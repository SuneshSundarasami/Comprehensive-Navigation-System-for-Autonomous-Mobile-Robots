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
from sensor_msgs.msg import Image

class AStarPathPlanner(Node):
    def __init__(self):
        super().__init__('astar_path_planner')

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.clearance_pub = self.create_publisher(OccupancyGrid, '/clearance_map', 10)


        self.map_data = None
        self.map_info = None
        self.current_pose = None
        self.end_pose = (4, -4)  # Goal in world coordinates
        self.graph = None
        self.clearance_map = None
        self.map_processed = False  # Flag to ensure map is only processed once

        self.get_logger().info('A* Path Planner Node initialized')

    def map_callback(self, msg):
        """Receives the map and constructs a graph for A*."""
        if self.map_processed:  # Skip processing if already done
            return
        
        self.map_info = msg.info
        raw_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        self.clearance_map = self.compute_clearance_map(raw_map)
        self.graph = self.create_graph(raw_map)
        self.map_processed = True  # Mark as processed

        self.get_logger().info(f'Received map: {msg.info.width}x{msg.info.height}, Resolution: {msg.info.resolution}')
        self.plan_path()  # Try planning a path if pose is available



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







    def create_graph(self, raw_map, obstacle_radius=3):
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
                        G.add_edge((x, y), neighbor, weight=1) 

        # Remove obstacles and add weights based on clearance
        for y in range(height):
            for x in range(width):
                if raw_map[y, x] == 100:  # Obstacle
                    # Remove the node and all neighboring nodes within the obstacle radius
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

        self.get_logger().info(f'Graph computed')
        return G

    def odom_callback(self, msg):
        """Receives odometry updates and triggers path planning."""
        self.current_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.map_processed:  # Only plan if the map has been processed
            self.plan_path()

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


    
    def plan_path(self):
        """Plans a path using A* with precomputed clearance."""
        if self.current_pose is None or not self.map_processed:
            return

        start_grid = self.world_to_grid(*self.current_pose)
        end_grid = self.world_to_grid(*self.end_pose)

        self.get_logger().info(f'Planning path from {start_grid} to {end_grid}')

        if start_grid not in self.graph or end_grid not in self.graph:
            self.get_logger().warn('Start or goal is in an invalid position!')
            return

        try:
            path_grid = nx.astar_path(self.graph, start_grid, end_grid, weight='weight', heuristic=self.clearance_aware_heuristic)
        except nx.NetworkXNoPath:
            self.get_logger().warn('No path found!')
            return

        important_path = self.extract_turns(path_grid)
        self.get_logger().info(f'Intial paths: {len(path_grid)} : Extracted paths:{len(important_path)}')
        path_msg = Path()
        path_msg.header.frame_id = 'odom'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for x, y in important_path:
            world_x, world_y = self.grid_to_world(x, y)
            pose = PoseStamped()
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info(f'Published path with {len(path_msg.poses)} poses')

    def clearance_aware_heuristic(self, a, b):
        """A* heuristic that favors paths with higher clearance using a fast lookup."""
        euclidean_dist = np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        clearance_factor = self.clearance_map[a[1], a[0]] 
        return euclidean_dist / (1.0 + (10**2) * clearance_factor)
    
    # def publish_clearance_map(self):
    #     """Publishes the precomputed clearance map as an OccupancyGrid message."""
    #     if self.clearance_map is None:
    #         self.get_logger().warn("Clearance map not available!")
    #         return

    #     max_clearance = np.max(self.clearance_map)
    #     normalized_clearance = (self.clearance_map / max_clearance) * 100  # Scale to 0-100

    #     clearance_msg = OccupancyGrid()
    #     clearance_msg.header.frame_id = "map"
    #     clearance_msg.header.stamp = self.get_clock().now().to_msg()
    #     clearance_msg.info = self.map_info
    #     clearance_msg.data = normalized_clearance.astype(np.int8).flatten().tolist()

    #     self.clearance_pub.publish(clearance_msg)
    #     self.get_logger().info(f'Clearance map published')




def main(args=None):
    rclpy.init(args=args)
    node = AStarPathPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
