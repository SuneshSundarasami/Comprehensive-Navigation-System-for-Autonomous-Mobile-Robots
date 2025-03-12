import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose2D
from visualization_msgs.msg import MarkerArray
from std_srvs.srv import Trigger
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
        
        # Initialize status client
        self.status_client = self.create_client(Trigger, 'get_pfield_status')
        while not self.status_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Status service not available, waiting...')
        self.status_request = Trigger.Request()
        
        # Create subscribers and publishers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10)
        self.goal_publisher = self.create_publisher(
            Pose2D, 
            'end_pose', 
             10)

        # Set up QoS profile for markers
        marker_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create visualization publisher with QoS
        self.visualization_pub = self.create_publisher(
            MarkerArray,
            'frontier_markers',
            marker_qos
        )

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
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
        self.executing = False
        self.current_goal = None
        self.last_status = ""
        
        # Add variables to store current frontiers
        self.current_frontiers = None
        self.current_selected_centroid = None
        
        # Add variables to store frontiers in map coordinates
        self.map_origin = None
        self.map_resolution = None
        
        # Create timer for status checking
        self.create_timer(0.5, self.check_status)

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

    def check_status(self):
        """Periodic status check"""
        if not self.executing:
            return
            
        future = self.status_client.call_async(self.status_request)
        future.add_done_callback(self.status_callback)

    def status_callback(self, future):
        """Handle status response"""
        try:
            response = future.result()
            status = response.message
            
            if status != self.last_status:
                self.last_status = status
                self.get_logger().info(f'Current status: {status}')
                
                if status in ['Goal Position Reached! Aligned orientation!', 'Waiting for goal pose']:
                    self.executing = False
                    self.current_goal = None
                    # Reset frontiers when goal is reached
                    self.current_frontiers = None
                    self.current_selected_centroid = None
                    
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    def get_robot_position(self):
        """Get robot position with proper transform handling"""
        try:
            # Get the latest available transform
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),  # Use latest available transform
                rclpy.duration.Duration(seconds=0.1)
            )
            
            self.robot_position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ])
            return True
            
        except Exception as e:
            self.get_logger().debug(f'Transform error: {str(e)}')
            return False

    def publish_goal(self, x, y, theta=1.0):
        """Publish a goal pose"""
        if self.executing:
            return False
            
        goal = Pose2D()
        goal.x = x
        goal.y = y
        goal.theta = theta
        
        self.goal_publisher.publish(goal)
        self.executing = True
        self.current_goal = [x, y]
        
        self.get_logger().info(f'Published goal: ({x:.2f}, {y:.2f}, {theta:.2f})')
        return True

    def map_callback(self, msg):
        """Process incoming map data and publish frontiers"""
        if not self.get_robot_position():
            return

        try:
            # Check if map parameters have changed
            current_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
            current_resolution = msg.info.resolution
            
            map_changed = (self.map_origin != current_origin or 
                          self.map_resolution != current_resolution)
            
            if map_changed:
                self.map_origin = current_origin
                self.map_resolution = current_resolution
                # Force frontier redetection when map parameters change
                self.current_frontiers = None
                self.current_selected_centroid = None
            
            # Convert map to numpy array
            self.latest_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
            
            # Always visualize current frontiers if they exist
            if self.current_frontiers is not None:
                markers = self.visualizer.create_frontier_markers(
                    self.current_frontiers,
                    msg.info,
                    self.robot_position,
                    self.current_selected_centroid
                )
                self.visualization_pub.publish(markers)
            
            # Detect new frontiers only if we don't have current ones or reached the goal
            if self.current_frontiers is None:
                frontier_points = self.frontier_detector.detect_frontiers(self.latest_map)
                if len(frontier_points) > 0:
                    self.current_frontiers = frontier_points
                    
                    # Select new goal only if not executing
                    if not self.executing:
                        selected_centroid, score = self.goal_selector.select_goal(
                            frontier_points,
                            self.latest_map,
                            msg.info,
                            self.robot_position,
                            self.previous_goals
                        )

                        if selected_centroid is not None:
                            self.current_selected_centroid = selected_centroid
                            
                            # Calculate and publish goal
                            x = selected_centroid[1] * msg.info.resolution + msg.info.origin.position.x
                            y = selected_centroid[0] * msg.info.resolution + msg.info.origin.position.y

                            if self.publish_goal(x, y):
                                self.previous_goals.append([x, y])
                                if len(self.previous_goals) > 10:
                                    self.previous_goals.pop(0)
                                self.get_logger().info(f'Selected frontier with score: {score:.2f}')
                    
        except Exception as e:
            self.get_logger().error(f'Error in map_callback: {str(e)}')
            
def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()