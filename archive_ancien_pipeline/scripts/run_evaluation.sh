#!/bin/bash
# Automatise N essais d'evaluation quantitative du modele BC, pour un goal donne.
# Usage : ./run_evaluation.sh [nombre_essais] [nom_goal]
# nom_goal parmi : G1 G2 G3 G4 G5 (defaut G1)
#
# Version renforcee : nettoyage plus robuste entre essais (delais plus longs,
# redemarrage du daemon ROS2, verification explicite avant de continuer)
# pour eviter que des residus d'un essai polluent le suivant sur WSL2.

set -e

N_TRIALS="${1:-15}"
GOAL_NAME="${2:-G1}"
WS_DIR="$HOME/imitanav_evaluation/ros2_ws"

declare -A GOAL_X_MAP=( [G1]=5.5 [G2]=6.5 [G3]=1.0 [G4]=6.0 [G5]=0.5 )
declare -A GOAL_Y_MAP=( [G1]=1.5 [G2]=-2.0 [G3]=2.0 [G4]=0.0 [G5]=-2.0 )

if [[ -z "${GOAL_X_MAP[$GOAL_NAME]}" ]]; then
    echo "Goal inconnu: $GOAL_NAME (attendu: G1 G2 G3 G4 G5)"
    exit 1
fi

export EVAL_GOAL_X="${GOAL_X_MAP[$GOAL_NAME]}"
export EVAL_GOAL_Y="${GOAL_Y_MAP[$GOAL_NAME]}"
export EVAL_RESULTS_CSV="$HOME/imitanav_evaluation/results/evaluation_${GOAL_NAME}.csv"

# Trace le modele utilise, pour eviter toute ambiguite sur quel modele a ete evalue
MODEL_USED="${BC_MODEL_PATH:-$HOME/imitanav_evaluation/models/bc_model.pt}"

source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"
export LIBGL_ALWAYS_SOFTWARE=1

mkdir -p "$(dirname "$EVAL_RESULTS_CSV")"
rm -f "$EVAL_RESULTS_CSV"

echo "=== Evaluation quantitative : $N_TRIALS essais sur $GOAL_NAME ($EVAL_GOAL_X, $EVAL_GOAL_Y) ==="
echo "=== Modele utilise : $MODEL_USED ==="

cleanup_full() {
    pkill -9 -f "gzserver" 2>/dev/null || true
    pkill -9 -f "gzclient" 2>/dev/null || true
    pkill -9 -f "ros2 launch" 2>/dev/null || true
    pkill -9 -f "spawn_entity" 2>/dev/null || true
    pkill -9 -f "spawner" 2>/dev/null || true
    pkill -9 -f "robot_state_publisher" 2>/dev/null || true
    pkill -9 -f "joint_state_publisher" 2>/dev/null || true
    pkill -9 -f "motion_control" 2>/dev/null || true
    pkill -9 -f "static_transform_publisher" 2>/dev/null || true
    pkill -9 -f "eval_trial_node" 2>/dev/null || true

    # Attente active : verifie qu'aucun noeud residuel ne traine, jusqu'a 15s
    for wait_attempt in $(seq 1 8); do
        remaining=$(ros2 node list 2>/dev/null | grep -c "create3\|motion_control\|robot_state" || true)
        if [[ "$remaining" -eq 0 ]]; then
            break
        fi
        sleep 2
    done

    # Redemarre le daemon ROS2 pour repartir sur une base DDS propre
    ros2 daemon stop > /dev/null 2>&1 || true
    sleep 1
    ros2 daemon start > /dev/null 2>&1 || true
    sleep 2
}

# Nettoyage initial avant le tout premier essai
cleanup_full

for i in $(seq 1 "$N_TRIALS"); do
    echo ""
    echo ">>> Essai $i / $N_TRIALS ($GOAL_NAME)"

    ros2 launch create3_lidar_description create3_lidar_full.launch.py \
        use_rviz:=false > /tmp/eval_sim_log_$i.txt 2>&1 &
    SIM_PID=$!

    echo "Attente du controller_manager..."
    controller_ready=false
    for attempt in $(seq 1 40); do
        if ros2 service list 2>/dev/null | grep -q "/controller_manager/list_controllers"; then
            echo "controller_manager pret."
            controller_ready=true
            break
        fi
        sleep 2
    done

    if [[ "$controller_ready" == false ]]; then
        echo "ECHEC: controller_manager jamais pret pour l'essai $i, on saute cet essai."
        kill -9 "$SIM_PID" 2>/dev/null || true
        cleanup_full
        continue
    fi

    echo "Attente du diffdrive_controller..."
    diffdrive_ready=false
    for attempt in $(seq 1 20); do
        if ros2 topic list 2>/dev/null | grep -q "/diffdrive_controller/cmd_vel_unstamped"; then
            echo "diffdrive_controller pret."
            diffdrive_ready=true
            break
        fi
        sleep 1
    done

    if [[ "$diffdrive_ready" == false ]]; then
        echo "ECHEC: diffdrive_controller jamais pret pour l'essai $i, on saute cet essai."
        kill -9 "$SIM_PID" 2>/dev/null || true
        cleanup_full
        continue
    fi

    # Verification supplementaire : le service controller_manager doit vraiment
    # repondre (pas seulement le topic present) avant de considerer le robot pilotable.
    # Un topic peut apparaitre avant que le service ne reponde vraiment sur WSL2.
    echo "Verification que le controller repond reellement..."
    controller_responds=false
    for attempt in $(seq 1 15); do
        if timeout 3 ros2 service call /controller_manager/list_controllers \
            controller_manager_msgs/srv/ListControllers "{}" > /tmp/eval_ctrl_check_$i.txt 2>&1; then
            if grep -q "diffdrive_controller" /tmp/eval_ctrl_check_$i.txt; then
                echo "Controller confirme actif."
                controller_responds=true
                break
            fi
        fi
        sleep 2
    done

    if [[ "$controller_responds" == false ]]; then
        echo "ECHEC: le service controller_manager ne repond pas correctement pour l'essai $i, on saute cet essai."
        kill -9 "$SIM_PID" 2>/dev/null || true
        cleanup_full
        continue
    fi

    # Laisse le temps a la simulation de vraiment se stabiliser (physique, capteurs)
    sleep 5

    # Applique le safety_override avec retry, car le node peut ne pas etre pret immediatement
    for retry in 1 2 3; do
        if ros2 param set /motion_control safety_override full > /dev/null 2>&1; then
            break
        fi
        sleep 2
    done

    python3 "$WS_DIR/src/create3_il/create3_il/eval_trial_node.py" "$i" \
        > /tmp/eval_trial_log_$i.txt 2>&1 || true

    kill -9 "$SIM_PID" 2>/dev/null || true
    cleanup_full
done

echo ""
echo "=== Termine. Resultats dans $EVAL_RESULTS_CSV ==="
echo "=== Modele evalue : $MODEL_USED ==="
echo "Lance maintenant : python3 ~/imitanav_evaluation/scripts/summarize_evaluation.py $EVAL_RESULTS_CSV"