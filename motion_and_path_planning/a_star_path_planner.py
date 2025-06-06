#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
import networkx as nx
from scipy.ndimage import distance_transform_edt
import cv2
from cv_bridge import CvBridge
import tf_transformations
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D

class AStarPathPlanner(Node):
    def __init__(self):
        super().__init__('astar_path_planner')

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.clearance_pub = self.create_publisher(OccupancyGrid, '/clearance_map', 10)

        # Add goal pose subscription
        self.goal_sub = self.create_subscription(
            Pose2D,
            'goal_pose',  # Match the topic name from end_pose_publisher
            self.goal_callback,
            10
        )

        # Initialize state variables with hardcoded end pose
        self.map_data = None
        self.map_info = None
        self.current_pose = None
        self.end_pose = (10000, 10000)  # Hardcoded initial end pose
        self.initial_path_planned = False  # Flag to track initial path
        self.last_end_pose = None
        self.graph = None
        self.clearance_map = None
        self.map_processed = False

        self.get_logger().info('A* Path Planner Node initialized')

    def map_callback(self, msg):
        """Receives the map and constructs a graph for A*."""
        if self.map_processed:  # Skip processing if already done
            return
        
        self.map_info = msg.info
        raw_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        self.clearance_map = self.compute_clearance_map(raw_map)
        self.graph = self.create_graph(raw_map)
        self.map_processed = True
    
        # Try planning initial path with hardcoded end pose
        if not self.initial_path_planned and self.current_pose:
            self.plan_path()
            self.initial_path_planned = True

    def compute_clearance_map(self, raw_map):
        try:
            # Convert occupancy grid to binary (0 = free, 1 = obstacle)
            obstacle_mask = (raw_map == 100)
            clearance_map = distance_transform_edt(~obstacle_mask)
            self.clearance_map = clearance_map


            np.savetxt('/home/sunesh/ros2_ws/src/amr_project_amr_t04/clearance_maps/clearance_map_swapped.csv',
                    np.transpose(self.clearance_map), delimiter=',', fmt='%.4f')

            self.get_logger().info("Swapped clearance map saved as a CSV file.")



            # Normalize for visualization
            max_clearance = np.max(clearance_map)
            clearance_img = (clearance_map / max_clearance * 255).astype(np.uint8) if max_clearance > 0 else np.zeros_like(clearance_map)

            # Apply a color map
            clearance_colormap = cv2.applyColorMap(clearance_img, cv2.COLORMAP_JET)
            if clearance_colormap is None or clearance_colormap.size == 0:
                self.get_logger().error("Error: Clearance colormap is empty.")
                return

            # Save the clearance map image
            save_path = '/home/sunesh/ros2_ws/src/amr_project_amr_t04/clearance_maps/clearance_map.png'
            if not cv2.imwrite(save_path, clearance_colormap):
                self.get_logger().error(f"Error: Could not save image at {save_path}.")
            else:
                self.get_logger().info(f"Clearance map image saved to {save_path}")

            # Create a raw map visualization (obstacles black, free space white)
            raw_map_vis = np.where(raw_map == 100, 0, 255).astype(np.uint8)
            raw_map_color = cv2.cvtColor(raw_map_vis, cv2.COLOR_GRAY2BGR)

            # Blend the images
            overlay = cv2.addWeighted(clearance_colormap, 0.7, raw_map_color, 0.3, 0)

            # Save the overlay image
            overlay_path = '/home/sunesh/ros2_ws/src/amr_project_amr_t04/clearance_maps/overlay_clearance_map.png'
            if not cv2.imwrite(overlay_path, overlay):
                self.get_logger().error(f"Error: Could not save overlay image at {overlay_path}.")
            else:
                self.get_logger().info(f"Overlay image saved to {overlay_path}")

            return clearance_map

        except Exception as e:
            self.get_logger().error(f"Exception in compute_clearance_map: {str(e)}")
            return None







    def create_graph(self, raw_map, obstacle_radius=2):
        """Creates a weighted graph representation of the grid map for A* with wider obstacles."""
        height, width = raw_map.shape
        G = nx.grid_2d_graph(width, height)

        for x in range(width):
            for y in range(height):
                if (x, y) not in G:  # Skip obstacles
                    continue
                # Check all 4 diagonal directions and add edges if valid
                for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                    neighbor = (x + dx, y + dy)
                    if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                        if (neighbor not in G):  # Skip obstacles
                            continue
                        G.add_edge((x, y), neighbor, weight=1.414)  # Diagonal cost

        # Remove obstacles and unexplored areas
        for y in range(height):
            for x in range(width):
                if raw_map[y, x] == 100 or raw_map[y, x] == -1:  # Obstacle or unexplored
                    # Remove the node and all neighboring nodes within the radius
                    for dx in range(-obstacle_radius, obstacle_radius + 1):
                        for dy in range(-obstacle_radius, obstacle_radius + 1):
                            nx_coord = x + dx
                            ny_coord = y + dy
                            if 0 <= nx_coord < width and 0 <= ny_coord < height:
                                if (nx_coord, ny_coord) in G:
                                    G.remove_node((nx_coord, ny_coord))
                else:
                    if (x, y) in G:
                        clearance = self.clearance_map[y, x]
                        G.nodes[(x, y)]['weight'] = 1 / clearance

        self.get_logger().info('Graph computed with unexplored areas removed')
        return G

    def odom_callback(self, msg):
        """Receives odometry updates."""
        new_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
        # Update current pose
        self.current_pose = new_pose
        
        # Only plan initial path when map is processed and we have current pose
        if (not self.initial_path_planned and self.map_processed and 
            self.current_pose and self.end_pose):
            self.plan_path()
            self.initial_path_planned = True
            self.last_end_pose = self.end_pose

    def world_to_grid(self, x, y):
        """Converts world coordinates to grid coordinates."""
        gx = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        gy = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        """Converts grid coordinates back to world coordinates."""
        x = gx * self.map_info.resolution + self.map_info.origin.position.x
        y = gy * self.map_info.resolution + self.map_info.origin.position.y
        return x, y




    def extract_turns(self, path_grid, angle_threshold=30, distance_threshold=1.0):
        """Extracts key waypoints (turns) from the path based on significant direction changes and distance."""
        key_points = []
        
        if len(path_grid) < 3:
            return path_grid  # Return the path as is if there are fewer than 3 points
        
        # Convert path to NumPy array for easier manipulation
        path_grid = np.array(path_grid)
        
        # Add the first point as a starting point
        key_points.append(tuple(path_grid[0]))

        # Calculate the differences between consecutive points (x, y)
        diffs = np.diff(path_grid, axis=0)
        
        # Calculate the angles of each segment in degrees
        angles = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0]))

        # Check for significant turns (angle difference above the threshold)
        for i in range(1, len(angles)):
            angle_diff = abs(angles[i] - angles[i - 1])
            angle_diff = np.mod(angle_diff + 180, 360) - 180  # Wrap angle difference to [-180, 180]
            
            # Calculate the distance between consecutive points
            distance = np.linalg.norm(path_grid[i] - path_grid[i - 1])
            
            # If the angle difference exceeds the threshold and the distance is large enough, it's a turn
            if abs(angle_diff) > angle_threshold and distance > distance_threshold:
                key_points.append(tuple(path_grid[i]))

        # Add the last point as the endpoint
        key_points.append(tuple(path_grid[-1]))
        
        return key_points

    def compute_waypoints(self, important_path):
        """Computes (x, y, θ) for the extracted path corners."""
        waypoints = []

        for i in range(len(important_path)):
            x, y = important_path[i]
            world_x, world_y = self.grid_to_world(x, y)

            if i < len(important_path) - 1:
                # Compute heading angle θ from current point to the next point
                next_x, next_y = important_path[i + 1]
                next_world_x, next_world_y = self.grid_to_world(next_x, next_y)
                theta = np.arctan2(next_world_y - world_y, next_world_x - world_x)
            else:
                # Keep last point's orientation the same as previous
                theta = waypoints[-1][2] if waypoints else 0.0

            waypoints.append((world_x, world_y, theta))

        return waypoints
    
    def find_closest_valid_point(self, target_grid, start_grid=None, max_search_radius=100):
        """Find the closest valid and reachable point to the target in the graph."""
        if target_grid in self.graph and (start_grid is None or 
            nx.has_path(self.graph, start_grid, target_grid)):
            return target_grid

        target_x, target_y = target_grid
        min_distance = float('inf')
        closest_point = None
        candidates = []

        # Search in expanding squares around the target
        for r in range(1, max_search_radius):
            # Check all points in the perimeter (including diagonals)
            for i in range(-r, r + 1):
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1),  # Cardinals
                            (1, 1), (1, -1), (-1, 1), (-1, -1)]:  # Diagonals
                    check_x = target_x + (i * dx)
                    check_y = target_y + (i * dy)
                    point = (check_x, check_y)
                    
                    if point in self.graph:
                        dist = np.sqrt((check_x - target_x)**2 + (check_y - target_y)**2)
                        # Store candidates sorted by distance
                        candidates.append((dist, point))

            # Sort candidates by distance and check reachability
            if candidates:
                candidates.sort(key=lambda x: x[0])  # Sort by distance
                for dist, point in candidates:
                    # Check if point is reachable from start
                    if start_grid is None or nx.has_path(self.graph, start_grid, point):
                        self.get_logger().debug(
                            f'Found reachable point at distance {dist:.2f} from target'
                        )
                        return point

        return None

    def plan_path(self):
        """Plans a path using A* with precomputed clearance."""
        if self.current_pose is None or not self.map_processed:
            return

        # Convert poses to grid coordinates
        start_grid = self.world_to_grid(*self.current_pose)
        end_grid = self.world_to_grid(*self.end_pose)
        
        self.get_logger().info(f'Planning path from {start_grid} to {end_grid}')
        
        # Debug graph information
        self.get_logger().info(f'Graph has {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges')
        
        # Find valid start point
        if start_grid not in self.graph:
            start_grid = self.find_closest_valid_point(start_grid)
            if start_grid is None:
                self.get_logger().warn('No valid start position found! Publishing direct path to goal.')
                if self.end_pose == (10000, 10000) :
                    return
                # Create and publish single-pose path to goal
                path_msg = Path()
                path_msg.header.frame_id = 'odom'
                path_msg.header.stamp = self.get_clock().now().to_msg()
                
                pose = PoseStamped()
                pose.pose.position.x = self.end_pose[0]
                pose.pose.position.y = self.end_pose[1]
                pose.pose.position.z = 0.0
                
                # Calculate orientation towards goal
                dx = self.end_pose[0] - self.current_pose[0]
                dy = self.end_pose[1] - self.current_pose[1]
                theta = np.arctan2(dy, dx)
                pose.pose.orientation = self.yaw_to_quaternion(theta)
                
                path_msg.poses.append(pose)
                self.path_pub.publish(path_msg)
                self.get_logger().info('Published direct path to goal')
                return
            self.get_logger().info(f'Adjusted start position to: {start_grid}')
        
        # Find valid end point
        if end_grid not in self.graph:
            end_grid = self.find_closest_valid_point(end_grid)
            if end_grid is None:
                self.get_logger().warn('No valid goal position found! Publishing direct path to goal.')
                if self.end_pose == (10000, 10000) :
                    return
                # Create and publish single-pose path to goal
                path_msg = Path()
                path_msg.header.frame_id = 'odom'
                path_msg.header.stamp = self.get_clock().now().to_msg()
                
                pose = PoseStamped()
                pose.pose.position.x = self.end_pose[0]
                pose.pose.position.y = self.end_pose[1]
                pose.pose.position.z = 0.0
                
                # Calculate orientation towards goal
                dx = self.end_pose[0] - self.current_pose[0]
                dy = self.end_pose[1] - self.current_pose[1]
                theta = np.arctan2(dy, dx)
                pose.pose.orientation = self.yaw_to_quaternion(theta)
                
                path_msg.poses.append(pose)
                self.path_pub.publish(path_msg)
                self.get_logger().info('Published direct path to goal')
                return
                
            self.get_logger().info(f'Adjusted goal position to: {end_grid}')
            # Update end_pose with the new valid position
            self.end_pose = self.grid_to_world(*end_grid)
        
        # Verify connectivity
        if not nx.has_path(self.graph, start_grid, end_grid):
            self.get_logger().error(f'No path exists between {start_grid} and {end_grid}')
            path_msg = Path()
            path_msg.header.frame_id = 'odom'
            path_msg.header.stamp = self.get_clock().now().to_msg()
            
            pose = PoseStamped()
            pose.pose.position.x = self.end_pose[0]
            pose.pose.position.y = self.end_pose[1]
            pose.pose.position.z = 0.0
            
            # Calculate orientation towards goal
            dx = self.end_pose[0] - self.current_pose[0]
            dy = self.end_pose[1] - self.current_pose[1]
            theta = np.arctan2(dy, dx)
            pose.pose.orientation = self.yaw_to_quaternion(theta)
            
            path_msg.poses.append(pose)
            self.path_pub.publish(path_msg)
            self.get_logger().info('Published direct path to goal')
            return

            return

        try:
            # Calculate path with increased weight on path length
            path_grid = nx.astar_path(
                self.graph, 
                start_grid, 
                end_grid,
                weight='weight',
                heuristic=lambda a, b: 1.5 * np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
            )
            
            self.get_logger().info(f'Found path with {len(path_grid)} points')
            
            # Extract key points and compute waypoints
            important_path = self.extract_turns(path_grid)
            waypoints = self.compute_waypoints(important_path)
            
            # Create and publish path message
            path_msg = Path()
            path_msg.header.frame_id = 'odom'
            path_msg.header.stamp = self.get_clock().now().to_msg()

            for world_x, world_y, theta in waypoints:
                pose = PoseStamped()
                pose.pose.position.x = world_x
                pose.pose.position.y = world_y
                pose.pose.position.z = 0.0
                pose.pose.orientation = self.yaw_to_quaternion(theta)
                path_msg.poses.append(pose)

            self.path_pub.publish(path_msg)
            self.get_logger().info(f'Published path with {len(path_msg.poses)} waypoints')

        except nx.NetworkXNoPath:
            self.get_logger().error('A* failed to find a path!')
            return
        except Exception as e:
            self.get_logger().error(f'Path planning failed: {str(e)}')
            return

    def yaw_to_quaternion(self, yaw):
        """Converts a yaw angle (theta) to a quaternion using tf transformations."""
        quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
        return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    
    def clearance_aware_heuristic(self, a, b):
        """A* heuristic that favors paths with higher clearance using a fast lookup."""
        euclidean_dist = np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        clearance_factor = self.clearance_map[a[1], a[0]] 
        return euclidean_dist / (1.0 + (10**2) * clearance_factor)
    
    def goal_callback(self, msg):
        """Handle new goal pose from end_pose_publisher."""
        new_end_pose = (msg.x, msg.y)
        
        # Only update if goal has changed and initial path was already planned
        if (self.initial_path_planned and new_end_pose != self.end_pose):
            self.get_logger().info(f'New goal received: {new_end_pose}')
            self.end_pose = new_end_pose
            if self.map_processed and self.current_pose:
                self.plan_path()
                self.last_end_pose = new_end_pose


def main(args=None):
    rclpy.init(args=args)
    node = AStarPathPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()