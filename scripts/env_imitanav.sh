#!/bin/bash
# Environnement ROS 2 ISOLE pour ce projet. A SOURCER (ne pas executer) :
#     source scripts/env_imitanav.sh
#
# Il ignore tous les autres workspaces colcon charges par ~/.bashrc
# (tunibot, cstam, create3_ws, stage_imitation_learning, ...) et ne garde que
# ROS 2 Humble + le workspace de CE projet. Il fonctionne meme si le terminal
# a deja ete pollue.

_IMITANAV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Retire des variables de chemin toute entree d'un workspace colcon place
# sous $HOME (.../install/...). Les chemins systeme et WSL ne sont pas touches.
_strip_ws() {
    echo "$1" | tr ':' '\n' | grep -v "^$HOME/.*/install\(/\|$\)" | paste -sd: -
}

unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH AMENT_CURRENT_PREFIX
unset GAZEBO_MODEL_PATH GAZEBO_RESOURCE_PATH GAZEBO_PLUGIN_PATH
export PYTHONPATH="$(_strip_ws "${PYTHONPATH:-}")"
export LD_LIBRARY_PATH="$(_strip_ws "${LD_LIBRARY_PATH:-}")"
export PATH="$(_strip_ws "$PATH")"

_had_u=0; [[ $- == *u* ]] && _had_u=1
set +u
[ -f /usr/share/gazebo/setup.sh ] && source /usr/share/gazebo/setup.sh
source /opt/ros/humble/setup.bash
if [ -f "$_IMITANAV_ROOT/ros2_ws/install/local_setup.bash" ]; then
    source "$_IMITANAV_ROOT/ros2_ws/install/local_setup.bash"
else
    echo "ATTENTION : workspace non compile ($_IMITANAV_ROOT/ros2_ws/install)"
fi
[ "$_had_u" -eq 1 ] && set -u

export ROS_LOCALHOST_ONLY=1
export LIBGL_ALWAYS_SOFTWARE=1
