from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='amr_project_amr_t04',
            executable='pfieldmodel',
            name='potential_field_node',
            output='screen'
        ),
        
        Node(
            package='amr_project_amr_t04',
            executable='astar',
            name='astar_planner_node',
            output='screen'
        ),
        
        Node(
            package='amr_project_amr_t04',
            executable='pose_exec',
            name='pose_executor_node',
            output='screen'
        ),
        
        Node(
            package='amr_project_amr_t04',
            executable='path_vis',
            name='path_visualizer_node',
            output='screen'
        )
    ])