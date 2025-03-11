from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    x_arg = DeclareLaunchArgument(
        'x',
        default_value='0.0',
        description='X coordinate of the goal pose'
    )
    
    y_arg = DeclareLaunchArgument(
        'y',
        default_value='0.0',
        description='Y coordinate of the goal pose'
    )
    
    theta_arg = DeclareLaunchArgument(
        'theta',
        default_value='0.0',
        description='Theta (orientation) of the goal pose'
    )

    return LaunchDescription([
        # Add the launch arguments
        x_arg,
        y_arg,
        theta_arg,
        
        # Nodes
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
            output='screen',
            parameters=[{
                'end_pose.x': LaunchConfiguration('x'),
                'end_pose.y': LaunchConfiguration('y'),
                'end_pose.theta': LaunchConfiguration('theta')
            }]
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