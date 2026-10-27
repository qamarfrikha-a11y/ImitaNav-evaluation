#!/bin/bash
# Reproduit le demarrage de simulation de run_campaign.sh et mesure QUAND
# /motion_control devient visible. Ne lance aucun essai.
#   ./scripts/test_sim_start.sh                      # config de la campagne
#   FORCE_LOCALONLY=0 ./scripts/test_sim_start.sh    # decouverte reseau normale

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source "$ROOT/scripts/env_imitanav.sh"
[[ -n "${FORCE_LOCALONLY:-}" ]] && export ROS_LOCALHOST_ONLY="$FORCE_LOCALONLY"
echo "ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-non defini}  RMW=${RMW_IMPLEMENTATION:-defaut}"

kill_all() {
    for p in gzserver gzclient "ros2 launch" spawn_entity spawner robot_state_publisher \
             joint_state_publisher motion_control irobot_create static_transform_publisher; do
        pkill -9 -f "$p" 2>/dev/null || true
    done
}

kill_all
sleep 3
ros2 daemon stop > /dev/null 2>&1 || true
sleep 1
ros2 daemon start > /dev/null 2>&1 || true
sleep 2
if pgrep -f "gzserver|motion_control" > /dev/null; then
    echo "ERREUR : il reste des processus Gazebo/Create3, abandon."
    exit 1
fi

T0=$SECONDS
ros2 launch create3_lidar_description create3_lidar_full.launch.py \
    use_rviz:=false spawn_dock:=true x:=-0.081 y:=0.007 yaw:=0.125 \
    > /tmp/test_sim.txt 2>&1 < /dev/null &
SIM=$!
echo "simulation lancee (log : /tmp/test_sim.txt)"

ok=0
for i in $(seq 1 25); do
    t=$((SECONDS - T0))
    n=$(timeout 20 ros2 node list 2>/dev/null | grep -c "motion_control")
    r=$(timeout 20 ros2 param set /motion_control safety_override full 2>&1 | head -c 80 | tr '\n' ' ')
    echo "t=${t}s  /motion_control dans node list: ${n}  |  param set: ${r}"
    if [[ "$r" == *successful* ]]; then ok=1; break; fi
    sleep 2
done

echo
if [[ "$ok" -eq 1 ]]; then echo "=> OK apres $((SECONDS - T0)) s"; else echo "=> ECHEC apres $((SECONDS - T0)) s"; fi
echo "--- ros2 node list ---";   timeout 20 ros2 node list 2>&1 | head -30
echo "--- ros2 daemon status ---"; ros2 daemon status 2>&1 | head -3
echo "--- /dev/shm (extrait) ---"; ls /dev/shm 2>/dev/null | head -8
echo "--- memoire ---"; free -h | head -2

kill_all
kill "$SIM" 2>/dev/null || true
