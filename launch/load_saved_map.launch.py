from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    map_path = '/home/sunesh/ros2_ws/src/amr_project_amr_t04/my_map'
    map_name = 'closed_walls_world_map'
    
    return LaunchDescription([
        # Static Transform Publisher for map to odom using YAML origin values
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tf',
            arguments=['0', '0',  '0', '0', '0', '0', 'map', 'odom']
        ),
        
        # Map Server Node
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': os.path.join(map_path, f'{map_name}.yaml'),
                'topic_name': 'map',
                'frame_id': 'map',
                'use_sim_time': False,
                'publish_period_sec': 1.0
            }]
        ),
        
        # Lifecycle Manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': ['map_server'],
                'bond_timeout': 2.0,
                'attempt_respawn': True
            }]
        ),
        

    ])