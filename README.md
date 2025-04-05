# Autonomous Mobile Robot (AMR) Project

Welcome to the AMR Project repository. This project is part of the Master of Autonomous Systems program at Hochschule Bonn-Rhein-Sieg.

## Project Overview

This project presents an integrated navigation framework for autonomous mobile robots, addressing the critical challenges of path planning, localization, and environment exploration on the Robile platform. We implement comprehensive solutions for efficient navigation in unknown environments while handling real-world constraints and sensor uncertainties.


## Key Features

### Path Planning
- Hybrid approach combining A* algorithm with a novel clearance-based heuristic
- Two-phase path simplification algorithm preserving critical turns
- Potential field control with emergency obstacle avoidance mechanisms (0.4m safety radius)
- Weighted zones that penalize proximity to obstacles

### Localization
- Particle filter localization using Monte Carlo methods
- Dual motion models (linear and circular) for different movement patterns
- Optimized sensor likelihood calculations using adaptive ray subsampling
- Ball tree spatial indexing for efficient distance queries
- Log-likelihood calculation for numerical stability

### Autonomous Exploration
- Frontier detection system using DBSCAN clustering
- Multi-criteria decision making that balances:
  - Information gain
  - Distance costs
  - Safety clearance threshold
- Real-time progress monitoring and visualization
- Adaptive blacklisting mechanism for problematic areas

## Technologies Used

- **Platform**: Robile robot
- **Framework**: ROS2 (Robot Operating System)
- **Libraries**: 
  - NetworkX (for graph-based path planning)
  - NumPy (for efficient numerical operations)
  - scikit-learn (for DBSCAN clustering)
  - Ball Tree (for spatial indexing)

## Repository Structure

```
amr-project-amr-t04/
│
├── src/
│   ├── path_planning/
│   │   ├── astar_planner.py        # A* algorithm with custom heuristic
│   │   ├── clearance_map.py        # Generates safety maps using distance transforms
│   │   ├── potential_field.py      # Motion control with obstacle avoidance
│   │   └── path_simplifier.py      # Two-phase path simplification algorithm
│   │
│   ├── localization/
│   │   ├── particle_filter.py      # Monte Carlo localization implementation
│   │   ├── motion_models.py        # Linear and circular motion models
│   │   ├── sensor_models.py        # Likelihood field and ray casting models
│   │   └── resampling.py           # Diversity-preserving resampling
│   │
│   ├── exploration/
│   │   ├── frontier_detection.py   # Identifies boundaries between known/unknown space
│   │   ├── cluster_analysis.py     # DBSCAN clustering for frontier analysis
│   │   ├── goal_selection.py       # Multi-criteria decision making
│   │   └── progress_monitor.py     # Tracks exploration completion
│   │
│   └── visualization/
│       ├── path_visualizer.py      # Renders planned paths in RViz
│       └── exploration_markers.py  # Visualizes frontier clusters
│
├── launch/
│   ├── navigation.launch.py        # Launch file for path planning and control
│   ├── localization.launch.py      # Launch file for particle filter
│   └── exploration.launch.py       # Launch file for autonomous exploration
│
├── config/
│   ├── path_planning_params.yaml   # Configuration for A* and potential field
│   ├── particle_filter_params.yaml # Configuration for localization
│   └── exploration_params.yaml     # Configuration for frontier exploration
│
├── docs/
│   └── AMR_Autonomous_navigation_exploration_localisation.pdf # Project report
│
└── README.md                       # Project readme file
```

## Getting Started

To get started with the project, follow these steps:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/HBRS-AMR/amr-project-amr-t04.git
   cd amr-project-amr-t04
   ```

2. **Install ROS2 and dependencies:**
   Ensure you have ROS2 installed. Then, install the dependencies:
   ```sh
   rosdep install --from-paths src --ignore-src -r -y
   ```

3. **Build the workspace:**
   ```sh
   colcon build
   source install/setup.bash
   ```

4. **Launch the navigation system:**
   ```sh
   ros2 launch amr_project_amr_t04 motionandpathplanner.launch.py  
   ```

5. **Launch the localization system:**
   ```sh
   ros2 launch amr_project_amr_t04 localizer.launch.py  
   ```

6. **Launch the exploration system:**
   ```sh
   ros2 run amr_project_amr_t04 frontier_explorer
   ```

## Technical Implementation Details

### Path Planning
The A* algorithm uses a custom heuristic function:
```
h(n) = d_euclidean · (1 + 100 · e^(-c/2)) + P_obstacle
```
where:
- c is the clearance value at position a
- P_obstacle is a progressive penalty based on clearance zones

### Localization
The particle filter implements motion models for both linear and circular movement patterns:

**Linear Motion Model:**
```
x_t = x_(t-1) + v_x·cos(θ_(t-1))·Δt - v_y·sin(θ_(t-1))·Δt + ϵ_x
y_t = y_(t-1) + v_x·sin(θ_(t-1))·Δt + v_y·cos(θ_(t-1))·Δt + ϵ_y
θ_t = θ_(t-1) + ω·Δt + ϵ_θ
```

**Circular Motion Model:**
Used when angular velocity is significant, for more accurate curved trajectory prediction.

### Exploration
The multi-criteria goal selection formula:
```
S_total = w_u · U + w_o · C - w_d · (D/D_max)
```
where:
- U is the count of unexplored cells within a given radius
- C is the clearance value [0, 1]
- D is the distance to the robot
- D_max is the maximum allowed distance
- w_u, w_o, and w_d are weight parameters

## Video Demonstrations

- [Localization Demo](https://youtu.be/Qq2m1GcVXMM)
- [Exploration and Navigation Demo](https://youtu.be/jIJaT5MY-9g)

## Future Work

- Improve clearance map generation efficiency
- Enhance localization for faster moving robots
- Integrate additional sensory modalities such as depth cameras
- Develop learning-based approaches for parameter adaptation

## References

1. H. Qin, S. Shao, T. Wang, X. Yu, Y. Jiang, and Z. Cao, "Review of autonomous path planning algorithms for mobile robots," Drones, vol. 7, no. 3, 2023.
2. D. Fox et al., "Monte carlo localization: Efficient position estimation for mobile robots," Proceedings of the National Conference on Artificial Intelligence, pp. 343–349, 1999.

## Acknowledgments

We acknowledge the support of teaching assistant Anudeep Sai Akula in setting up the Robile platform and guiding us in case of any issues.
