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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D

class AStarPathPlanner(Node):
    def __init__(self):
        super().__init__('astar_path_planner')

        map_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
    )

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
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
        self.end_pose = (5, -4)  # Hardcoded initial end pose
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
        
        # Clear robot footprint from occupancy grid
        raw_map = self.clear_robot_footprint(raw_map)
        
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

    def clear_robot_footprint(self, raw_map):
        """Clear robot footprint from occupancy grid based on URDF dimensions."""
        try:

            robot_width = 0.233*2.5  # meters
            robot_length = 0.233*3.5  # meters
            
            # Convert robot dimensions to grid cells
            robot_width_cells = int(robot_width / self.map_info.resolution)
            robot_length_cells = int(robot_length / self.map_info.resolution)
            
            # Get robot position in grid coordinates
            if self.current_pose:
                robot_x, robot_y = self.world_to_grid(*self.current_pose)
                
                # Calculate footprint bounds
                half_width = robot_width_cells // 2
                half_length = robot_length_cells // 2
                
                # Clear the area around robot position
                for dx in range(-half_width, half_width + 1):
                    for dy in range(-half_length, half_length + 1):
                        grid_x = robot_x + dx
                        grid_y = robot_y + dy
                        
                        # Check if within map bounds
                        if (0 <= grid_x < raw_map.shape[1] and 
                            0 <= grid_y < raw_map.shape[0]):
                            raw_map[grid_y, grid_x] = 0  # Mark as free space
                
                self.get_logger().debug(
                    f'Cleared robot footprint: {robot_width_cells}x{robot_length_cells} cells'
                )
                
            return raw_map
            
        except Exception as e:
            self.get_logger().error(f'Error clearing robot footprint: {str(e)}')
            return raw_map

    def create_graph(self, raw_map, obstacle_radius=1):
        """Creates a weighted graph representation of the grid map for A*."""
        height, width = raw_map.shape
        G = nx.grid_2d_graph(width, height)
        
        # Add diagonal edges without weights
        for x in range(width):
            for y in range(height):
                if (x, y) not in G:  # Skip obstacles
                    continue
                # Add diagonal connections
                for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                    neighbor = (x + dx, y + dy)
                    if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                        if (neighbor not in G):  # Skip obstacles
                            continue
                        G.add_edge((x, y), neighbor)  # No weight specified for diagonal edges

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

        self.get_logger().info(f'Graph created with {len(G.nodes)} nodes and {len(G.edges)} edges')
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
    
    def find_closest_valid_point(self, target_grid, max_search_radius=100):
        """Find the closest valid point to the target in the graph."""
        if target_grid in self.graph:
            return target_grid

        target_x, target_y = target_grid
        min_distance = float('inf')
        closest_point = None

        # Search in expanding squares around the target
        for r in range(1, max_search_radius):
            for dx in range(-r, r + 1):
                for dy in [-r, r]:  # Top and bottom edges
                    check_x, check_y = target_x + dx, target_y + dy
                    point = (check_x, check_y)
                    if point in self.graph:
                        dist = np.sqrt((check_x - target_x)**2 + (check_y - target_y)**2)
                        if dist < min_distance:
                            min_distance = dist
                            closest_point = point

                for dy in range(-r + 1, r):  # Left and right edges
                    for dx in [-r, r]:
                        check_x, check_y = target_x + dx, target_y + dy
                        point = (check_x, check_y)
                        if point in self.graph:
                            dist = np.sqrt((check_x - target_x)**2 + (check_y - target_y)**2)
                            if dist < min_distance:
                                min_distance = dist
                                closest_point = point

            if closest_point is not None:
                break

        return closest_point

    def find_alternative_goal(self, start_grid, end_grid, max_attempts=20):
        """Find an alternative reachable goal point when direct path is not possible."""
        self.get_logger().info(f'Searching for alternative goal near {end_grid}')
        
        original_x, original_y = end_grid
        candidates = []
        
        # Search in expanding circles
        for radius in range(1, max_attempts + 1):
            found_in_radius = False
            points_in_radius = []
            
            # More points for larger radii
            num_points = 16 * radius
            for angle in np.linspace(0, 2*np.pi, num_points):
                # Calculate point on circle
                x = int(original_x + radius * np.cos(angle))
                y = int(original_y + radius * np.sin(angle))
                test_point = (x, y)
                
                # Check if point is valid and reachable
                if test_point in self.graph and nx.has_path(self.graph, start_grid, test_point):
                    dist_to_goal = np.sqrt((x - original_x)**2 + (y - original_y)**2)
                    # Get path length to this point
                    path_length = len(nx.astar_path(self.graph, start_grid, test_point))
                    
                    points_in_radius.append({
                        'point': test_point,
                        'dist_to_goal': dist_to_goal,
                        'path_length': path_length
                    })
                    found_in_radius = True
            
            if found_in_radius:
                # Sort points in this radius by distance to goal
                points_in_radius.sort(key=lambda x: x['dist_to_goal'])
                # Take the 3 closest points to goal from this radius
                candidates.extend(points_in_radius[:3])
                
                # If we have enough candidates, break
                if len(candidates) >= 5:
                    break
        
        if candidates:
            # Sort candidates by a weighted combination of goal distance and path length
            candidates.sort(key=lambda x: x['dist_to_goal'] * 2.0 + x['path_length'] * 0.1)
            best_point = candidates[0]['point']
            
            self.get_logger().info(
                f'Found alternative goal at {best_point}:\n'
                f'- Distance to original goal: {candidates[0]["dist_to_goal"]:.2f}\n'
                f'- Path length: {candidates[0]["path_length"]}'
            )
            return best_point
        
        self.get_logger().error('No alternative goals found!')
        return None

    def plan_path(self):
        """Plans a path using A* with clearance-aware heuristic and alternative goals."""
        if self.current_pose is None or not self.map_processed:
            return

        try:
            # Convert poses to grid coordinates
            start_grid = self.world_to_grid(*self.current_pose)
            end_grid = self.world_to_grid(*self.end_pose)
            
            self.get_logger().info(f'Planning path from {start_grid} to {end_grid}')
            
            # Find valid start point
            if start_grid not in self.graph:
                start_grid = self.find_closest_valid_point(start_grid)
                if start_grid is None:
                    self.get_logger().error('No valid start position found!')
                    return
                self.get_logger().info(f'Adjusted start position to: {start_grid}')
            
            # Find valid end point
            if end_grid not in self.graph:
                end_grid = self.find_closest_valid_point(end_grid)
                if end_grid is None:
                    self.get_logger().error('No valid goal position found!')
                    return
                self.get_logger().info(f'Adjusted goal position to: {end_grid}')
            
            # Check if path exists
            if not nx.has_path(self.graph, start_grid, end_grid):
                self.get_logger().warn('No direct path exists, searching for alternative goal...')
                alternative_goal = self.find_alternative_goal(start_grid, end_grid)
                
                if alternative_goal is None:
                    self.get_logger().error('No alternative path found!')
                    return
                    
                end_grid = alternative_goal
                self.end_pose = self.grid_to_world(*end_grid)
                self.get_logger().info(f'Using alternative goal: {end_grid}')
            
            # Use clearance-aware heuristic for A* search
            path_grid = nx.astar_path(
                self.graph, 
                start_grid, 
                end_grid,
                weight='weight',
                heuristic=self.clearance_aware_heuristic
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
            self.get_logger().info(
                f'Published path with {len(path_msg.poses)} waypoints\n'
                f'Path favors areas with higher clearance from obstacles'
            )

        except nx.NetworkXNoPath:
            self.get_logger().error('A* failed to find a path!')
        except Exception as e:
            self.get_logger().error(f'Path planning failed: {str(e)}')

    def yaw_to_quaternion(self, yaw):
        """Converts a yaw angle (theta) to a quaternion using tf transformations."""
        quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
        return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    
    def clearance_aware_heuristic(self, a, b):
        """A* heuristic that heavily penalizes paths near obstacles."""
        # Base distance calculation
        euclidean_dist = np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        
        try:
            # Get clearance at current position
            clearance = self.clearance_map[a[1], a[0]]
            max_clearance = np.max(self.clearance_map)
            
            # Exponential penalty for low clearance
            clearance_factor = np.exp(-clearance/2.0)  # Exponential decay
            
            # Progressive penalty zones
            if clearance < 2.0:  # Very close to obstacles
                obstacle_penalty = 10000.0
            elif clearance < 3.0:  # Near obstacles
                obstacle_penalty = 5000.0
            elif clearance < 4.0:  # Moderately close
                obstacle_penalty = 2500.0
            else:  # Safe distance
                obstacle_penalty = 0.0
            
            # Combined cost heavily favoring clear paths
            return euclidean_dist * (1.0 + 100*  clearance_factor) + obstacle_penalty
            
        except Exception as e:
            self.get_logger().warn(f'Heuristic calculation failed: {str(e)}, using basic distance')
            return euclidean_dist
    
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