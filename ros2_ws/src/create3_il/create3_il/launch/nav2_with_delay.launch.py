from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os

def generate_launch_description():
    # Arguments
    map_arg = DeclareLaunchArgument(
        'map',
        default_value=os.path.expanduser('~/imitanav_evaluation/maps/couloir_L.yaml'),
        description='Full path to map yaml file to load'
    )
    
    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.expanduser('~/imitanav_evaluation/config/nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    # Nœud map_server (démarré en premier)
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{'yaml_filename': LaunchConfiguration('map'), 
                     'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # Nœud AMCL (démarré après un délai)
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        parameters=[LaunchConfiguration('params_file')],
        output='screen'
    )

    # Lancer le reste de Nav2 (lifecycle_manager, planner, controller, etc.)
    nav2_bringup = Node(
        package='nav2_bringup',
        executable='bringup_launch.py',
        parameters=[LaunchConfiguration('params_file')],
        arguments=['map:=', LaunchConfiguration('map'), 
                   'use_sim_time:=', LaunchConfiguration('use_sim_time')],
        output='screen'
    )

    return LaunchDescription([
        map_arg,
        params_arg,
        use_sim_time_arg,
        
        # 1. Démarrer map_server
        map_server,
        
        # 2. Attendre 5 secondes pour que la carte soit chargée
        TimerAction(
            period=5.0,
            actions=[amcl]
        ),
        
        # 3. Démarrer le reste de Nav2
        TimerAction(
            period=7.0,
            actions=[nav2_bringup]
        )
    ])