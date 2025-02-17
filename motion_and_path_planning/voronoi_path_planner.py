import rclpy
from rclpy.node import Node
import numpy as np
# import cv2
from scipy.spatial import Voronoi
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import networkx as nx


class VoronoiPathPlanner(Node):
    def __init__(self):
        super().__init__('voronoi_path_planner')
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.grid = None

    def map_callback(self, msg):
        """ Convert occupancy grid to binary map and compute Voronoi diagram """
        width, height = msg.info.width, msg.info.height
        data = np.array(msg.data).reshape((height, width))
        binary_map = np.where(data > 50, 1, 0)  # Thresholding occupancy grid (1=obstacle, 0=free)

        # Compute Voronoi
        voronoi = self.compute_voronoi(binary_map)

        # Find the optimal path (example: using A* on Voronoi edges)
        start = (0, 0)  # Replace with actual robot position
        goal = (4, -4)   # Replace with actual goal position
        path = self.astar_on_voronoi(voronoi, start, goal)

        # Publish the path
        self.publish_path(path, msg.info)

    def compute_voronoi(self, grid):
        """ Compute the Voronoi diagram of free space """
        free_points = np.column_stack(np.where(grid == 0))
        vor = Voronoi(free_points)
        return vor

    def astar_on_voronoi(self, voronoi, start, goal):
        """ Perform A* search on Voronoi graph to find an obstacle-free path """

        # Create a NetworkX graph from Voronoi edges
        G = nx.Graph()
        
        # Add Voronoi vertices as graph nodes
        for i, vertex in enumerate(voronoi.vertices):
            if np.any(vertex < 0):  # Ignore out-of-bounds vertices
                continue
            G.add_node(i, pos=(vertex[0], vertex[1]))

        # Add edges between Voronoi ridge points
        for ridge in voronoi.ridge_vertices:
            if -1 in ridge:  # Skip ridges at infinity
                continue
            G.add_edge(ridge[0], ridge[1], weight=np.linalg.norm(voronoi.vertices[ridge[0]] - voronoi.vertices[ridge[1]]))

        # Find nearest Voronoi node to the start and goal
        start_node = min(G.nodes, key=lambda i: np.linalg.norm(voronoi.vertices[i] - start))
        goal_node = min(G.nodes, key=lambda i: np.linalg.norm(voronoi.vertices[i] - goal))

        # Run A* search on the Voronoi graph
        try:
            path_nodes = nx.astar_path(G, start_node, goal_node, weight='weight')
            path = [tuple(voronoi.vertices[i]) for i in path_nodes]
        except nx.NetworkXNoPath:
            self.get_logger().error("No path found!")
            path = [start, goal]  # Default to direct path if no solution

        return path

    def publish_path(self, path, map_info):
        """ Convert the computed path to a ROS Path message """
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for x, y in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x * map_info.resolution
            pose.pose.position.y = y * map_info.resolution
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info("Published path on /planned_path")

def main(args=None):
    rclpy.init(args=args)
    node = VoronoiPathPlanner()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
