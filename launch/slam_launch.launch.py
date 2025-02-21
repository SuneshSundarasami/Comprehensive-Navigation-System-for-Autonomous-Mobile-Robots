from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="slam_toolbox",
            executable="sync_slam_toolbox_node",
            name="slam_toolbox",
            parameters=["~/ros2_ws/src/amr_project_amr_t04/config/slam_toolbox_config.yaml"],
            output="screen",
        )
    ])
