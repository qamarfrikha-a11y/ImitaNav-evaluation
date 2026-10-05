#!/bin/bash
# Lance les essais d'un plan, un par un, avec reprise automatique.
#
# Usage : ./run_campaign.sh <plan.csv> <resultats.csv> [nb_max_essais]
#   ex. : ./run_campaign.sh results/experiments/plan_pilot.csv results/experiments/trials_pilot.csv
#
# - saute les essais dont le trial_uid est deja dans resultats.csv (reprise)
# - relance un essai rate (Gazebo qui plante...) une fois, puis le note dans
#   failed_trials.csv a cote du fichier de resultats
# - chaque essai redemarre Gazebo (etat propre, plus lent mais robuste)
# - DRY_RUN=1 : affiche les essais sans rien lancer
#
# Variables optionnelles (exportees avant l'appel) :
#   EVAL_USE_SAFETY=1|0  EVAL_USE_FINAL_APPROACH=1|0  EVAL_COLLISION_MODE=first|blocking
#   EVAL_TIMEOUT_S=160   EVAL_WALL_LIMIT=900 (limite en temps reel par essai)

set -u

PLAN="$(realpath -m "${1:?Usage: $0 <plan.csv> <resultats.csv> [nb_max]}")"
RESULTS="$(realpath -m "${2:?Usage: $0 <plan.csv> <resultats.csv> [nb_max]}")"
MAX_NEW="${3:-1000000}"
DRY_RUN="${DRY_RUN:-0}"
WALL_LIMIT="${EVAL_WALL_LIMIT:-900}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS_DIR="$ROOT/ros2_ws"
NODE="$WS_DIR/src/create3_il/create3_il/eval_campaign_node.py"
OUTDIR="$(dirname "$RESULTS")"
LOGDIR="$OUTDIR/logs"
FAILED="$OUTDIR/failed_trials.csv"
mkdir -p "$LOGDIR"

[[ -f "$PLAN" ]] || { echo "Plan introuvable : $PLAN"; exit 1; }
if [[ "$DRY_RUN" != "1" ]]; then
    [[ -f "$NODE" ]] || { echo "Noeud introuvable : $NODE"; exit 1; }
    [[ -f "$WS_DIR/install/setup.bash" ]] || { echo "Workspace non compile : $WS_DIR/install"; exit 1; }
    # Environnement ISOLE : ignore les autres workspaces charges par ~/.bashrc
    set +u
    source "$ROOT/scripts/env_imitanav.sh"
    set -u
    {
        echo "date: $(date -Iseconds)"
        echo "git_commit: $(git -C "$ROOT" rev-parse HEAD 2>/dev/null)"
        echo "AMENT_PREFIX_PATH:"; echo "$AMENT_PREFIX_PATH" | tr ':' '\n' | sed 's/^/  /'
        echo "irobot_create_description: $(ros2 pkg prefix irobot_create_description 2>&1)"
        dpkg -l 2>/dev/null | grep -i "ros-humble-irobot" | awk '{print "  " $2, $3}'
    } > "$OUTDIR/environment.txt"
fi

GIT_COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
SIM_PID=""

cleanup_full() {
    [[ -n "$SIM_PID" ]] && kill -9 "$SIM_PID" 2>/dev/null || true
    for p in gzserver gzclient "ros2 launch" spawn_entity spawner robot_state_publisher \
             joint_state_publisher motion_control static_transform_publisher eval_campaign_node; do
        pkill -9 -f "$p" 2>/dev/null || true
    done
    for _ in $(seq 1 8); do
        remaining=$(ros2 node list 2>/dev/null | grep -c "create3\|motion_control\|robot_state" || true)
        [[ "$remaining" -eq 0 ]] && break
        sleep 2
    done
    ros2 daemon stop > /dev/null 2>&1 || true
    sleep 1
    ros2 daemon start > /dev/null 2>&1 || true
    sleep 2
    SIM_PID=""
}

wait_until() {   # wait_until <essais> <pause_s> <commande>
    local n="$1" s="$2" cmd="$3"
    for _ in $(seq 1 "$n"); do
        if eval "$cmd"; then return 0; fi
        sleep "$s"
    done
    return 1
}

start_sim() {    # start_sim <uid> <x> <y> <yaw>
    local uid="$1" x="$2" y="$3" yaw="$4"
    ros2 launch create3_lidar_description create3_lidar_full.launch.py \
        use_rviz:=false spawn_dock:="${EVAL_SPAWN_DOCK:-true}" x:="$x" y:="$y" yaw:="$yaw" \
        > "$LOGDIR/${uid}_sim.txt" 2>&1 < /dev/null &
    SIM_PID=$!
    wait_until 40 2 'ros2 service list 2>/dev/null | grep -q "/controller_manager/list_controllers"' || return 1
    wait_until 20 1 'ros2 topic list 2>/dev/null | grep -q "/diffdrive_controller/cmd_vel_unstamped"' || return 1
    wait_until 15 2 'timeout 3 ros2 service call /controller_manager/list_controllers controller_manager_msgs/srv/ListControllers "{}" 2>&1 | grep -q diffdrive_controller' || return 1
    sleep 5
    for _ in 1 2 3; do
        ros2 param set /motion_control safety_override full > /dev/null 2>&1 && break
        sleep 2
    done
    return 0
}

TOTAL=$(( $(wc -l < "$PLAN") - 1 ))
echo "=== Campagne : $TOTAL essais dans le plan | commit $GIT_COMMIT ==="
echo "=== Resultats : $RESULTS ==="

n_new=0; n_skip=0; n_fail=0; idx=0
while IFS=, read -r uid exp goal gx gy method model pct tseed trseed sx sy syaw mpath <&3; do
    [[ -z "$uid" ]] && continue
    mpath="${mpath%$'\r'}"          # securite : retire un eventuel retour chariot
    idx=$((idx + 1))
    if [[ -f "$RESULTS" ]] && grep -q "^${uid}," "$RESULTS"; then
        n_skip=$((n_skip + 1)); continue
    fi
    if [[ "$n_new" -ge "$MAX_NEW" ]]; then break; fi

    echo ""
    echo ">>> [$idx/$TOTAL] $uid | $model seed$tseed | $goal | depart ($sx,$sy,$syaw)"
    if [[ "$DRY_RUN" == "1" ]]; then n_new=$((n_new + 1)); continue; fi

    MODEL_ABS="$ROOT/$mpath"
    if [[ ! -f "$MODEL_ABS" ]]; then
        echo "ECHEC : modele introuvable $MODEL_ABS"
        echo "$uid,model_missing,$(date -Iseconds)" >> "$FAILED"
        n_fail=$((n_fail + 1)); continue
    fi

    export BC_MODEL_PATH="$MODEL_ABS"
    export EVAL_RESULTS_CSV="$RESULTS"
    export EVAL_GOAL_X="$gx" EVAL_GOAL_Y="$gy" EVAL_GOAL_NAME="$goal"
    export EVAL_EXPERIMENT="$exp" EVAL_METHOD="$method" EVAL_MODEL_ID="$model"
    export EVAL_PCT_CORR="$pct" EVAL_TRAIN_SEED="$tseed" EVAL_TRIAL_SEED="$trseed"
    export EVAL_START_X="$sx" EVAL_START_Y="$sy" EVAL_START_YAW="$syaw"
    export EVAL_GIT_COMMIT="$GIT_COMMIT"

    ok=0; reason="inconnu"
    for attempt in 1 2; do
        cleanup_full
        if ! start_sim "$uid" "$sx" "$sy" "$syaw"; then
            reason="simulation_non_prete"
            echo "   tentative $attempt : $reason"
            continue
        fi
        timeout "$WALL_LIMIT" python3 "$NODE" "$uid" \
            > "$LOGDIR/${uid}_node.txt" 2>&1 < /dev/null
        rc=$?
        if grep -q "^${uid}," "$RESULTS" 2>/dev/null; then ok=1; break; fi
        case "$rc" in
            3) reason="robot_immobile" ;;
            4) reason="pas_de_donnees_capteurs" ;;
            5) reason="simulation_figee" ;;
            124) reason="limite_temps_reel" ;;
            *) reason="noeud_sans_resultat_rc${rc}" ;;
        esac
        echo "   tentative $attempt : $reason (voir $LOGDIR/${uid}_node.txt)"
    done
    cleanup_full

    if [[ "$ok" -eq 1 ]]; then
        n_new=$((n_new + 1))
        tail -n 1 "$RESULTS" | awk -F, '{print "   -> " $17 " | temps sim " $21 " s | traj " $23 " m"}'
    else
        echo "$uid,$reason,$(date -Iseconds)" >> "$FAILED"
        n_fail=$((n_fail + 1))
        echo "   ECHEC definitif : $reason (note dans $FAILED)"
    fi
done 3< <(tail -n +2 "$PLAN")

echo ""
echo "=== Termine : nouveaux=$n_new, deja faits=$n_skip, echecs=$n_fail ==="
