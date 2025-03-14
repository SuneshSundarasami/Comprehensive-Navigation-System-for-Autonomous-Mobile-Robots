from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='amr_project_amr_t04',
            executable='particle_filter',
            name='particle_filter',
            parameters=[{
                'num_particles': 100,
                'motion_noise': [0.1, 0.1, 0.1],
                'measurement_noise': 0.1,
                'resample_threshold': 0.5
            }]
        ),
        Node(
            package='amr_project_amr_t04',
            executable='particle_filter_visualizer',
            name='particle_filter_visualizer'
        )
    ])