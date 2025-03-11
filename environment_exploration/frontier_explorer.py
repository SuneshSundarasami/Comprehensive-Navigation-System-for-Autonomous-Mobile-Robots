import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
import numpy as np
import tf2_ros
from utils.visualization import FrontierVisualizer
from utils.frontier_detector import FrontierDetector
from utils.goal_selector import GoalSelector

class FrontierExplorationNode(Node):
    def __init__(self):
        super().__init__('frontier_exploration_node')
        
        # Initialize parameters
        self._init_parameters()
        
        # Create subscribers and publishers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10)
        self.goal_publisher = self.create_publisher(
            PoseStamped, 
            'goal', 
            10)
        self.visualization_pub = self.create_publisher(
            MarkerArray,
            'frontier_markers',
            10)

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initialize utility classes
        self.visualizer = FrontierVisualizer()
        self.frontier_detector = FrontierDetector(
            self.min_frontier_size,
            self.clustering_eps
        )
        self.goal_selector = GoalSelector(
            self.information_radius,
            self.min_distance,
            self.max_distance
        )

        # Initialize state variables
        self.robot_position = np.array([0.0, 0.0])
        self.previous_goals = []
        self.latest_map = None

    def _init_parameters(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                ('min_frontier_size', 5),
                ('clustering_eps', 2.0),
                ('information_radius', 10),
                ('min_distance', 0.5),
                ('max_distance', 5.0)
            ])

        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        self.clustering_eps = self.get_parameter('clustering_eps').value
        self.information_radius = self.get_parameter('information_radius').value
        self.min_distance = self.get_parameter('min_distance').value
        self.max_distance = self.get_parameter('max_distance').value

    def get_robot_position(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())
            self.robot_position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ])
            return True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warning(f'Failed to get robot position: {str(e)}')
            return False

    def map_callback(self, msg):
        if not self.get_robot_position():
            return

        self.latest_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        frontier_points = self.frontier_detector.detect_frontiers(self.latest_map)

        if len(frontier_points) > 0:
            selected_centroid, score = self.goal_selector.select_goal(
                frontier_points,
                self.latest_map,
                msg.info,
                self.robot_position,
                self.previous_goals
            )

            if selected_centroid is not None:
                # Visualize frontiers
                markers = self.visualizer.create_frontier_markers(
                    frontier_points,
                    msg.info,
                    self.robot_position,
                    selected_centroid
                )
                self.visualization_pub.publish(markers)

                # Create and publish goal
                goal = PoseStamped()
                goal.header.frame_id = "map"
                goal.header.stamp = self.get_clock().now().to_msg()
                goal.pose.position.x = selected_centroid[1] * msg.info.resolution + msg.info.origin.position.x
                goal.pose.position.y = selected_centroid[0] * msg.info.resolution + msg.info.origin.position.y
                goal.pose.orientation.w = 1.0

                self.previous_goals.append([goal.pose.position.x, goal.pose.position.y])
                if len(self.previous_goals) > 10:
                    self.previous_goals.pop(0)

                self.goal_publisher.publish(goal)
                self.get_logger().info(
                    f'Published goal at ({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f}) with score {score:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()