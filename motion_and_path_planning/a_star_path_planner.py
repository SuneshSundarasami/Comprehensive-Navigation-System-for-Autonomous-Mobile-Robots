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
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point, Quaternion, PoseStamped, TransformStamped

class AStarPathPlanner(Node):
    def __init__(self):
        super().__init__('astar_path_planner')

        # Initialize state variables first
        self.map_data = None
        self.map_info = None
        self.current_pose = None
        self.end_pose = None  # Changed from (1000, 1000) to None
        self.initial_path_planned = False
        self.graph = None
        self.clearance_map = None
        self.map_processed = False
        self.latest_map = None

        # Set up QoS profile
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # Create subscribers and publishers
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.clearance_pub = self.create_publisher(OccupancyGrid, '/clearance_map', 10)
        self.goal_sub = self.create_subscription(Pose2D, 'goal_pose', self.goal_callback, 10)

        # Add clearance map subscription
        self.clearance_sub = self.create_subscription(
            OccupancyGrid,
            '/clearance_map',
            self.clearance_callback,
            10
        )

        # Add transform buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Add timer for pose updates
        self.create_timer(0.1, self.update_robot_pose)  # 10Hz updates

        self.get_logger().info('A* Path Planner Node initialized')

    def map_callback(self, msg):
        """Receives the map and constructs a graph for A*."""
        if self.map_processed:  # Skip processing if already done
            return
        
        self.map_info = msg.info
        self.latest_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        raw_map = self.latest_map.copy()
        
        # Clear robot footprint from occupancy grid
        raw_map = self.clear_robot_footprint(raw_map)
        
        self.clearance_map = self.compute_clearance_map(raw_map)
        self.graph = self.create_graph(raw_map)
        self.map_processed = True

        # Try planning initial path with hardcoded end pose
        if not self.initial_path_planned and self.current_pose:
            self.plan_path()
            self.initial_path_planned = True

    def clearance_callback(self, msg):
        """Handle incoming clearance map"""
        self.clearance_map = np.array(msg.data).reshape(
            (msg.info.height, msg.info.width)
        ) / 100.0  # Convert back to float
        
        # Process map if we haven't yet
        if not self.map_processed and self.latest_map is not None:
            self.process_map()

    def process_map(self):
        """Process map and create graph once we have both map and clearance data"""
        if self.clearance_map is None:
            return
            
        raw_map = self.latest_map.copy()
        raw_map = self.clear_robot_footprint(raw_map)
        self.graph = self.create_graph(raw_map)
        self.map_processed = True
        
        # Try planning initial path
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

            robot_width = 0.233*5.5  # meters
            robot_length = 0.233*6.5  # meters
            
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

    def update_robot_pose(self):
        """Get current pose from tf transform"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            
            # Extract position from transform
            new_pose = (
                transform.transform.translation.x,
                transform.transform.translation.y
            )
            
            # Update current pose
            self.current_pose = new_pose
            
            # Plan initial path when ready
            if (not self.initial_path_planned and self.map_processed and 
                self.current_pose and self.end_pose):
                self.plan_path()
                self.initial_path_planned = True
                
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            if not hasattr(self, '_last_tf_error_time') or \
               (self.get_clock().now() - self._last_tf_error_time).nanoseconds / 1e9 > 5.0:
                self.get_logger().warn(f'Transform lookup failed: {str(e)}')
                self._last_tf_error_time = self.get_clock().now()

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
    
    def find_closest_valid_point(self, target_grid, max_search_radius=20):
        """Find the closest valid and reachable point to target."""
        if target_grid in self.graph:
            return target_grid

        target_x, target_y = target_grid
        height, width = self.latest_map.shape
        best_point = None
        min_dist = float('inf')

        # Search in expanding squares
        for r in range(1, max_search_radius + 1):
            points_found = False
            
            # Check all points in current radius
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    # Only check points on the perimeter
                    if abs(dx) == r or abs(dy) == r:
                        check_x = target_x + dx
                        check_y = target_y + dy
                        
                        # Validate map bounds
                        if not (0 <= check_x < width and 0 <= check_y < height):
                            continue
                        
                        point = (check_x, check_y)
                        
                        # Check if point is in graph (free space)
                        if point in self.graph:
                            dist = np.sqrt((check_x - target_x)**2 + (check_y - target_y)**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_point = point
                                points_found = True
            
            # If we found valid points in this radius, return the best one
            if points_found:
                self.get_logger().info(
                    f'Found valid point:'
                    f'\n- Original target: {target_grid}'
                    f'\n- Selected point: {best_point}'
                    f'\n- Distance: {min_dist:.2f} cells'
                )
                return best_point

        self.get_logger().warn(
            f'No valid point found within {max_search_radius} cells of {target_grid}'
        )
        return None

    def find_alternative_goal(self, start_grid, end_grid, max_path_points=5):
        """Find alternative goal by assuming unexplored regions are free."""
        try:
            self.get_logger().info(
                f'[Path Planner] Finding alternative goal:'
                f'\n- Start: {start_grid}'
                f'\n- Target: {end_grid}'
            )

            # Create temporary graph including unexplored space
            temp_map = self.latest_map.copy()
            temp_map[temp_map == -1] = 0  # Treat unexplored as free
            temp_graph = self.create_graph(temp_map)

            if not (start_grid in temp_graph and end_grid in temp_graph):
                self.get_logger().warn('Start or end point not in temporary graph')
                return None

            # Find path in temporary graph
            try:
                full_path = nx.astar_path(
                    temp_graph,
                    start_grid,
                    end_grid,
                    weight='weight',
                    heuristic=self.clearance_aware_heuristic
                )
            except nx.NetworkXNoPath:
                self.get_logger().warn('No path found in temporary graph')
                return None

            # Walk along path until we hit unexplored or obstacle
            valid_path = []
            for point in full_path:
                y, x = point[1], point[0]  # Convert to map coordinates
                
                # Check if point is in known free space
                if self.latest_map[y, x] == 0:  # Known free space
                    valid_path.append(point)
                else:  # Hit obstacle or unexplored
                    break

                # Stop if we have enough points
                if len(valid_path) >= max_path_points:
                    break

            if valid_path:
                # Take the last valid point as alternative goal
                alt_goal = valid_path[-10]
                self.get_logger().info(
                    f'[Path Planner] Found alternative goal:'
                    f'\n- Original target: {end_grid}'
                    f'\n- Alternative goal: {alt_goal}'
                    f'\n- Path length: {len(valid_path)}'
                )
                return alt_goal
            else:
                self.get_logger().warn('No valid points found in path')
                return None

        except Exception as e:
            self.get_logger().error(f'Error finding alternative goal: {str(e)}')
            return None

    def plan_path(self):
        """Plans a path using A* directly in map frame with enhanced error handling"""
        try:
            # Check prerequisites
            if self.current_pose is None:
                self.get_logger().warn('Current pose not available yet')
                return
            if not self.map_processed:
                self.get_logger().warn('Map not processed yet')
                return
            if not self.graph:
                self.get_logger().error('Graph not initialized')
                return
                
            # Log current state
            self.get_logger().info(
                f'Planning path with:'
                f'\n- Map processed: {self.map_processed}'
                f'\n- Graph nodes: {len(self.graph.nodes)}'
                f'\n- Current pose: {self.current_pose}'
                f'\n- End pose: {self.end_pose}'
            )

            # Convert poses to grid coordinates
            start_grid = self.world_to_grid(*self.current_pose)
            end_grid = self.world_to_grid(*self.end_pose)
            
            self.get_logger().info(
                f'Grid coordinates:'
                f'\n- Start grid: {start_grid}'
                f'\n- End grid: {end_grid}'
            )
            
            # Validate start point
            if start_grid not in self.graph:
                self.get_logger().warn(f'Start point {start_grid} not in graph, finding closest valid point')
                start_grid = self.find_closest_valid_point(start_grid)
                if not start_grid:
                    self.get_logger().error('No valid start position found!')
                    return
                self.get_logger().info(f'Using alternative start point: {start_grid}')
            
            # Validate end point
            if end_grid not in self.graph:
                self.get_logger().warn(f'End point {end_grid} not in graph, finding closest valid point')
                end_grid = self.find_closest_valid_point(end_grid)
                if not end_grid:
                    self.get_logger().error('No valid goal position found!')
                    return
                self.get_logger().info(f'Using alternative end point: {end_grid}')
            
            # Check path existence
            if not nx.has_path(self.graph, start_grid, end_grid):
                self.get_logger().warn('Direct path not possible, trying alternative goal...')
                alt_goal = self.find_alternative_goal(start_grid, end_grid)
                if not alt_goal:
                    self.get_logger().error('No alternative path possible!')
                    return
                end_grid = alt_goal
                self.get_logger().info(f'Using alternative goal: {end_grid}')
            
            # Find path
            self.get_logger().info('Computing A* path...')
            path_grid = nx.astar_path(
                self.graph, 
                start_grid, 
                end_grid,
                weight='weight',
                heuristic=self.clearance_aware_heuristic
            )
            
            if not path_grid:
                self.get_logger().error('A* failed to find a path!')
                return
                
            self.get_logger().info(f'Found path with {len(path_grid)} points')
            
            # Rest of the path processing...
            important_path = self.extract_turns(path_grid)
            waypoints = self.compute_waypoints(important_path)
            
            # Create and publish path message in map frame
            path_msg = Path()
            path_msg.header.frame_id = 'map'  # Always use map frame
            path_msg.header.stamp = self.get_clock().now().to_msg()

            for world_x, world_y, theta in waypoints:
                pose = PoseStamped()
                pose.pose.position.x = world_x
                pose.pose.position.y = world_y
                pose.pose.position.z = 0.0
                pose.pose.orientation = self.yaw_to_quaternion(theta)
                path_msg.poses.append(pose)

            # Publish path directly in map frame
            self.path_pub.publish(path_msg)

            self.get_logger().info(
                f'Published path with {len(path_msg.poses)} waypoints in map frame'
            )

        except Exception as e:
            import traceback
            self.get_logger().error(f'Path planning failed: {str(e)}')
            self.get_logger().error(traceback.format_exc())

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
        try:
            new_end_pose = (msg.x, msg.y)
            self.get_logger().info(f'Received goal: ({msg.x:.2f}, {msg.y:.2f})')
            
            # Update end pose
            self.end_pose = new_end_pose
            
            # Check if we can plan
            if not self.map_processed:
                self.get_logger().warn('Map not processed yet, storing goal for later')
                return
                
            if not self.current_pose:
                self.get_logger().warn('Current pose not available yet, storing goal for later')
                return
                
            if not self.graph:
                self.get_logger().warn('Graph not initialized yet, storing goal for later')
                return
                
            # Convert goal to grid coordinates and validate
            end_grid = self.world_to_grid(*new_end_pose)
            if not (0 <= end_grid[0] < self.latest_map.shape[1] and 
                    0 <= end_grid[1] < self.latest_map.shape[0]):
                self.get_logger().error('Goal position outside map bounds!')
                return
                
            self.get_logger().info('Starting path planning to new goal...')
            self.plan_path()

        except Exception as e:
            self.get_logger().error(f'Error in goal_callback: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def check_direct_path(self, start, end, raw_map):
        """Check if there's a clear path between two points using raw map."""
        x0, y0 = start
        x1, y1 = end
        
        # Get points along the line using numpy's linspace
        num_points = max(abs(x1 - x0), abs(y1 - y0)) * 2
        x_coords = np.linspace(x0, x1, num=int(num_points))
        y_coords = np.linspace(y0, y1, num=int(num_points))
        
        # Round to nearest integer for grid coordinates
        x_coords = np.round(x_coords).astype(int)
        y_coords = np.round(y_coords).astype(int)
        
        # Check each point along the line
        for x, y in zip(x_coords, y_coords):
            # Check if point is within map bounds
            if not (0 <= y < raw_map.shape[0] and 0 <= x < raw_map.shape[1]):
                return False
            # Check if point is obstacle
            if raw_map[y, x] == 100:
                return False
        return True

def main(args=None):
    rclpy.init(args=args)
    node = AStarPathPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()