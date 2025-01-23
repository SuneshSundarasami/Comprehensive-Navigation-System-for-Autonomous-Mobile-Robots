# AMR Project

## Project Objectives

The objective of this project is that you deploy some of the functionalities that were discussed during the course on a real robot platform. In particular, we want to have functionalities for path and motion planning, localisation, and environment exploration on the robot.

We will particularly use the Robile platform during the project; you are already familiar with this robot from the simulation you have been using throughout the semester as well as from the few practical lab sessions that we have had.

## Task Description

The project consists of three parts that are building on each other: (i) path and motion planning, (ii) localisation, and (iii) environment exploration.

### 1. Path and Motion Planning

You have already implemented a *potential field planner* in one of your assignments. In this first part of the project, you need to port your implementation to the real robot and ensure that it is working as well as it was in the simulated environment so that you can navigate towards global goals while avoiding obstacles. Then, integrate your potential field planner with a global path planner, namely first use a path planner (e.g. A*) to find a rough global trajectory of waypoints that the robot can follow to reach a goal and then use the potential field planner to navigate between the waypoints. This will make your potential field planner applicable to large environments, where it can navigate given an environment map.

### 2. Localisation

In one of the course lectures, we discussed Monte Carlo localisation as a practical solution to the robot localisation problem in an existing map. In this second part of the project, your objective is to implement your very own particle filter that you then integrate on the Robile. You should implement the simple version of the filter that we discussed in the lecture; however, if you have time and interest, you are free to additionally explore extensions / improvements to the algorithm, for example in the form of the adaptive Monte Carlo approach that we mentioned in the lecture.

### 3. Environment Exploration

The final objective of the project is to incorporate an environment exploration functionality to the robot. This will have to be combined with a SLAM component, namely you will need your exploration component to select poses to explore and a SLAM component that will take care of actually creating a map. The exploration algorithm should ideally select poses at the map fringe (i.e. poses that are at the boundary between the explored and unexplored region), but you are free to explore different pose selection strategies in your implementation.

# Practical Notes and Assumptions
* For the first two tasks in this project, we need an environment map to be given. For this purpose, you should use an already existing SLAM approach in ROS (such as the `slam_toolbox` that you also used to map simulated environments) to create a map of the environment where you conduct your tests.
* In the first task of the project, you will use a grid map to find a global task plan; however, this plan will be too granular to be integrated with the potential field planner, as every grid cell on the path will be considered an intermediate goal. To improve this, you need to post-process your path so that you extract a number of representative waypoints along the path; these will then be the intermediate goals of your potential field planner. How exactly you decide to post-process the path is up to you; for instance, you can take every n-th cell along the path as a waypoint (where n is predefined), or you can develop a smarter strategy and extract waypoints at important points along the path (e.g. sharp points).
* You should also use an existing SLAM approach for the last part of the project, such that this will need to run in parallel with the exploration component. The selection of poses should thus be done with respect to the most up-to-date map provided by the SLAM algorithm.

## Submission Guidelines

Your submission should be a short PDF report (maximum five pages using the following template: https://github.com/a2s-institute/rnd-project-report), where you briefly describe your approach for the different tasks. In the report, make sure to include:
* A URL of a repository with all the code that you have developed during the project. Make sure that the repository contains a README file to explain the contents of the repository and provide short usage guidelines.
* A short section that describes how each team member contributed to the project.
* URLs to videos demonstrating your developed functionalities on the real Robile platform (you can upload these videos anywhere, for example to Google Drive or YouTube). In the videos, make sure that you explicitly show that you are executing your components!

The report should be uploaded to LEA before the submission deadline. The grading of the project will be done on the basis of this submission.

## Demonstration

After the submission deadline, each group will also need to present their results in a live demonstration. We will agree on a date for the demonstration at a later date. The live demonstration does not count towards the project grade; it is just there so that you get some live demo experience and so that we can discuss any concrete issues that you have faced in your implementations.
