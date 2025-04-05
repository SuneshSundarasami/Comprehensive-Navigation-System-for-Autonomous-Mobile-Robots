from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'amr_project_amr_t04'
motion_planner_module = 'motion_and_path_planning'
exploration_module = 'environment_exploration'
localizer_module = 'localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add this line to include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Add config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sunesh',
    maintainer_email='sunesh@outlook.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'pfieldmodel = {motion_planner_module}.potential_field_implementation:main',
            f'pfieldmodel2 = {motion_planner_module}.pfield_v2:main',
            f'astar = {motion_planner_module}.a_star_path_planner:main',
            f'pose_exec = {motion_planner_module}.pose_execuetor:main',
            f'path_vis = {motion_planner_module}.path_visualizer:main',
            # Add frontier explorer entry point
            f'frontier_explorer = {exploration_module}.frontier_explorer:main',
            f'particle_filter = {localizer_module}.particle_filter:main',
            f'particle_filter_visualizer = {localizer_module}.visualize_particle_filter:main',
        ],
    },
)