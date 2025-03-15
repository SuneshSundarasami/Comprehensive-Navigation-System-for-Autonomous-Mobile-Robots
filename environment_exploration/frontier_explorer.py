import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist, Pose2D
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
import numpy as np
import tf2_ros
from .utils.visualization import FrontierVisualizer
from .utils.frontier_detector import FrontierDetector
from .utils.goal_selector import GoalSelector

class FrontierExplorationNode(Node):
    def __init__(self):
        super().__init__('frontier_exploration_node')
        
        # Simple parameters
        self.min_frontier_size = 5
        self.clustering_eps = 2.0
        self.max_distance = 5.0
        
        # Publishers
        self.goal_publisher = self.create_publisher(
            Pose2D,
            'goal_pose',
            10)
        self.visualization_pub = self.create_publisher(
            MarkerArray,
            'frontier_markers',
             10)
             
        # Add cmd_vel publisher for spinning
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10)

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Add progress subscriber
        self.progress_sub = self.create_subscription(
            String,
            '/pose_progress',
            self.progress_callback,
            10
        )
        
        # Initialize utilities
        self.visualizer = FrontierVisualizer()
        self.frontier_detector = FrontierDetector(
            self.min_frontier_size,
            self.clustering_eps
        )
        self.goal_selector = GoalSelector(
            information_radius=5.0,
            min_distance=0.5,
            max_distance=5.0,
            logger=self.get_logger()
        )

        # State variables
        self.robot_position = np.array([0.0, 0.0])
        self.previous_goals = []
        self.latest_map = None
        self.executing = False
        self.waiting_for_completion = False
        
        # Add spin control variables
        self.is_spinning = False
        self.spin_start_time = None
        self.spin_duration = 6.0  # Time to complete full rotation (seconds)
        self.spin_timer = None

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
        except Exception as e:
            self.get_logger().warning(f'Failed to get robot position: {str(e)}')
            return False

    def progress_callback(self, msg):
        """Handle pose execution progress"""
        self.get_logger().info(f'Progress update: {msg.data}')
        
        if "All poses completed!" in msg.data:
            self.detect_and_publish_frontier()


    def detect_and_publish_frontier(self):
        """Detect and publish new frontier goal"""
            
        try:
            # Get current frontiers
            frontier_points = self.frontier_detector.detect_frontiers(self.latest_map)

            if len(frontier_points) > 0:
                # Select best frontier based on current position
                selected_centroid, distance_score = self.goal_selector.select_goal(
                    frontier_points,
                    self.latest_map,
                    self.latest_map_info,
                    self.robot_position,
                    []  # Empty list instead of previous_goals
                )

                if selected_centroid is not None:
                    # Visualize current frontiers
                    markers = self.visualizer.create_frontier_markers(
                        frontier_points,
                        self.latest_map_info,
                        self.robot_position,
                        selected_centroid
                    )
                    self.visualization_pub.publish(markers)

                    # Publish new goal
                    goal = Pose2D()
                    goal.x = selected_centroid[1] * self.latest_map_info.resolution + self.latest_map_info.origin.position.x
                    goal.y = selected_centroid[0] * self.latest_map_info.resolution + self.latest_map_info.origin.position.y
                    goal.theta = 1.0

                    self.goal_publisher.publish(goal)
                    self.executing = True
                    self.get_logger().info(
                        f'Published new frontier goal at ({goal.x:.2f}, {goal.y:.2f}) at distance {-distance_score:.2f}m'
                    )

        except Exception as e:
            self.get_logger().error(f'Error detecting frontier: {str(e)}')

    def map_callback(self, msg):
        # if not self.get_robot_position() or self.executing or self.waiting_for_completion:
        #     return

        try:
            # Store map data and info
            self.latest_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
            self.latest_map_info = msg.info
            
            # Only detect frontiers if not executing or waiting
            if not self.executing and not self.waiting_for_completion:
                self.detect_and_publish_frontier()
                
        except Exception as e:
            self.get_logger().error(f'Error in map_callback: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()