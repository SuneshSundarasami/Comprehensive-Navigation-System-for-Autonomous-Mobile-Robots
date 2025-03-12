from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true')
    
    # Create our own slam params file
    slam_params_file = os.path.join(get_package_share_directory('amr_project_amr_t04'),
                                  'config', 'slam_params.yaml')
    
    start_slam_toolbox_node = Node(
        parameters=[
            slam_params_file,
            {'use_sim_time': use_sim_time},
            {'odom_frame': 'odom'},
            {'base_frame': 'base_link'},
            {'map_frame': 'map'},
            {'map_update_interval': 1.0},
            {'resolution': 0.05},
            {'max_laser_range': 15.0},
            {'minimum_time_interval': 0.2},
            {'transform_timeout': 0.2},
            {'tf_buffer_duration': 30.0},
            {'stack_size_to_use': 40000000},
            {'enable_interactive_mode': False},
            {'minimum_travel_distance': 0.1},
            {'minimum_travel_heading': 0.1},
            {'scan_buffer_size': 10},
            {'link_match_minimum_response_fine': 0.1},
            {'link_scan_maximum_distance': 5.0},
            {'loop_search_maximum_distance': 3.0},
            {'do_loop_closing': True},
            {'loop_match_minimum_chain_size': 3},
            {'loop_match_minimum_response_coarse': 0.35},
            {'loop_match_minimum_response_fine': 0.45},
            {'occupancy_grid_publish_period': 1.0},
            {'publish_occupancy_map': True},
            {'free_thresh': 0.196},
            {'occupied_thresh': 0.65},
            {'map_start_pose': [0.0, 0.0, 0.0]},
            {'track_unknown_space': True},
            {'use_pose_extrapolator': True},
            {'debug_visualization': False},
            {'ceres_loss_function': 'HuberLoss'}
        ],
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen')

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(start_slam_toolbox_node)
    
    return ld