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
from .utils.exploration_monitor import ExplorationProgress

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
        
        # self.visualizer.set_logger(self.get_logger())
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
        self.goal_selector.initialize_clearance_subscriber(self)
        self.visualizer = FrontierVisualizer(self.get_logger())

        # State variables
        self.robot_position = np.array([0.0, 0.0])
        self.previous_goals = []
        self.latest_map = None
        self.executing = False
        self.waiting_for_completion = False
        
        # Add spin control variables
        self.is_spinning = False
        self.spin_start_time = None
        self.spin_duration = 100.0  # Time to complete full rotation (seconds)
        self.spin_timer = None

        # Add timeout parameters
        self.goal_timeout = 120.0  # 2 minutes timeout
        self.goal_start_time = None
        self.create_timer(1.0, self.check_goal_timeout)  # Check timeout every second

        # Add movement timeout parameters
        self.movement_timeout = 10.0  # 10 seconds timeout for no movement
        self.last_movement_time = self.get_clock().now()
        self.create_timer(1.0, self.check_movement_timeout)
        
        # Subscribe to cmd_vel to monitor movement
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Add timeout tracking
        self.consecutive_timeouts = 0
        self.max_consecutive_timeouts = 2

        # Add movement timeout tracking
        self.consecutive_movement_timeouts = 0
        self.max_movement_timeouts = 2
        self.movement_timeout = 5.0  # 5 seconds timeout for no movement

        self.last_position = None  # Add this to track the last position
        self.position_threshold = 1  # Threshold for position change (in meters)

        # Add position update timer
        self.create_timer(0.1, self.update_position)  # Update position at 10Hz

        # Add exploration progress monitor
        self.progress_monitor = ExplorationProgress(self.get_logger())
        self.create_timer(1.0, self.check_mapping_progress)

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
        
        if "Starting execution of" in msg.data:
            # Reset movement timer when starting new path
            self.last_movement_time = self.get_clock().now()
            # Extract number of poses from the message
            num_poses = int(msg.data.split()[3])
            self.get_logger().info(
                f'\nPath Execution Started:'
                f'\n- Total poses in path: {num_poses}'
                f'\n- Current robot position: ({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f})'
                f'\n- Status: Beginning path traversal'
            )
        elif "Completed pose" in msg.data:
            # Reset movement timer when completing poses
            self.last_movement_time = self.get_clock().now()
            # Extract current pose and total poses
            current, total = map(int, msg.data.split()[2].split('/'))
            self.get_logger().info(
                f'\nPath Progress Update:'
                f'\n- Completed pose: {current} of {total}'
                f'\n- Progress: {(current/total)*100:.1f}%'
                f'\n- Current position: ({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f})'
            )
        elif "All poses completed!" in msg.data:
            self.get_logger().info(
                f'\nPath Execution Completed:'
                f'\n- Final position: ({self.robot_position[0]:.2f}, {self.robot_position[1]:.2f})'
                f'\n- Status: Ready for next frontier'
            )
            self.executing = False
            self.goal_start_time = None  # Reset timeout
            self.detect_and_publish_frontier()

    def detect_and_publish_frontier(self):
        """Detect and publish new frontier goal with visualization"""
        try:
            if not self.get_robot_position():
                self.get_logger().warn('Could not get robot position')
                return

            if self.latest_map is None:
                self.get_logger().warn('No map data available')
                return

            # Detect frontiers
            frontier_points = self.frontier_detector.detect_frontiers(self.latest_map)
            self.get_logger().info(f'Detected {len(frontier_points)} frontier points')
            
            if len(frontier_points) > 0:
                # Select goal
                selected_point, score, centroids, labels = self.goal_selector.select_goal(
                    frontier_points,
                    self.latest_map,
                    self.latest_map_info,
                    self.robot_position,
                    self.previous_goals
                )

                if selected_point is not None:
                    # Update progress monitor with selected cluster points
                    if labels is not None:
                        selected_mask = labels == labels[np.where(np.all(frontier_points == selected_point, axis=1))[0][0]]
                        self.progress_monitor.update_cluster(frontier_points[selected_mask])

                    # Create and publish visualization markers
                    markers = self.visualizer.create_frontier_markers(
                        frontier_points,
                        self.latest_map_info,
                        self.robot_position,
                        selected_point,
                        centroids,
                        labels
                    )
                    
                    # Log marker information
                    self.get_logger().info(
                        f'Publishing markers:'
                        f'\n- Number of markers: {len(markers.markers)}'
                        f'\n- Selected point: ({selected_point[0]}, {selected_point[1]})'
                        f'\n- Number of frontiers: {len(frontier_points)}'
                    )
                    
                    # Publish markers
                    self.visualization_pub.publish(markers)

                    # Convert to world coordinates and publish goal
                    goal = Pose2D()
                    goal.x = float(selected_point[1] * self.latest_map_info.resolution + 
                                 self.latest_map_info.origin.position.x)
                    goal.y = float(selected_point[0] * self.latest_map_info.resolution + 
                                 self.latest_map_info.origin.position.y)
                    goal.theta = 0.0

                    # Store goal and publish
                    self.previous_goals.append((goal.x, goal.y))
                    if len(self.previous_goals) > 10:
                        self.previous_goals.pop(0)

                    self.goal_publisher.publish(goal)
                    self.executing = True
                    self.goal_start_time = self.get_clock().now()

                    self.get_logger().info(
                        f'\nPublished new frontier goal:'
                        f'\n- Position: ({goal.x:.2f}, {goal.y:.2f})'
                        f'\n- Score: {score:.3f}'
                    )
                else:
                    self.get_logger().warn('No valid goal point selected')
            else:
                self.get_logger().info('No frontier points detected')

        except Exception as e:
            self.get_logger().error(f'Error in detect_and_publish_frontier: {str(e)}')

    def select_closest_frontier(self, frontier_points):
        """Select the closest frontier point to the robot."""
        if frontier_points is None or len(frontier_points) == 0:
            return None
            
        # Convert robot position to grid coordinates
        robot_x = int((self.robot_position[0] - self.latest_map_info.origin.position.x) 
                      / self.latest_map_info.resolution)
        robot_y = int((self.robot_position[1] - self.latest_map_info.origin.position.y) 
                      / self.latest_map_info.resolution)
        robot_grid = np.array([robot_y, robot_x])  # Note: grid coordinates are (y,x)
        
        # Find closest frontier point
        distances = np.linalg.norm(frontier_points - robot_grid, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = frontier_points[closest_idx]
        
        return closest_point

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

    def check_goal_timeout(self):
        """Check if current goal has timed out by monitoring position"""
        if self.executing and self.goal_start_time is not None:
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.goal_start_time).nanoseconds / 1e9
            
            if elapsed_time > self.goal_timeout:
                # Check if position has changed
                current_position = np.array([self.robot_position[0], self.robot_position[1]])
                
                if self.last_position is None:
                    self.last_position = current_position
                    return
                
                # Calculate distance moved
                distance_moved = np.linalg.norm(current_position - self.last_position)
                
                if distance_moved < self.position_threshold:
                    self.consecutive_timeouts += 1
                    self.get_logger().warn(
                        f'Goal timed out after {elapsed_time:.1f} seconds!'
                        f'\n- Robot hasn\'t moved: distance = {distance_moved:.3f}m'
                        f'\n- Current position: ({current_position[0]:.2f}, {current_position[1]:.2f})'
                        f'\n- Previous position: ({self.last_position[0]:.2f}, {self.last_position[1]:.2f})'
                        f'\n- Consecutive timeouts: {self.consecutive_timeouts}'
                    )
                    self.executing = False
                    self.goal_start_time = None
                    
                    # Stop the robot
                    stop_cmd = Twist()
                    self.cmd_vel_pub.publish(stop_cmd)
                    
                    # Find new frontier
                    self.detect_and_publish_frontier()
                else:
                    # Position has changed significantly, reset timeout
                    self.goal_start_time = current_time
                    self.get_logger().debug(
                        f'Robot is making progress:'
                        f'\n- Distance moved: {distance_moved:.3f}m'
                        f'\n- Resetting timeout timer'
                    )
                
                # Update last position
                self.last_position = current_position

    def cmd_vel_callback(self, msg):
        """Monitor robot movement commands"""
        if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
            self.last_movement_time = self.get_clock().now()

    def check_movement_timeout(self):
        """Check if robot hasn't moved for too long"""
        if self.executing:
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.last_movement_time).nanoseconds / 1e9
            
            if elapsed_time > self.movement_timeout:
                self.consecutive_movement_timeouts += 1
                self.get_logger().warn(
                    f'Robot inactive for {elapsed_time:.1f} seconds! '
                    f'Consecutive movement timeouts: {self.consecutive_movement_timeouts}'
                )
                # Stop the robot
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                
                # Reset states
                self.executing = False
                self.goal_start_time = None
                self.last_movement_time = current_time
                
                # Find new frontier
                self.detect_and_publish_frontier()

    def update_position(self):
        """Continuously update and monitor robot position"""
        if self.get_robot_position():  # Using existing get_robot_position method
            current_position = np.array([self.robot_position[0], self.robot_position[1]])
            
            if self.last_position is not None:
                # Calculate distance moved
                distance_moved = np.linalg.norm(current_position - self.last_position)
                
                if distance_moved > 0.01:  # Only log if moved more than 1cm
                    self.get_logger().debug(
                        f'Robot moved {distance_moved:.3f}m:'
                        f'\n- From: ({self.last_position[0]:.2f}, {self.last_position[1]:.2f})'
                        f'\n- To: ({current_position[0]:.2f}, {current_position[1]:.2f})'
                    )
            
            self.last_position = current_position

    def check_mapping_progress(self):
        """Check if current cluster area is sufficiently mapped"""
        if self.executing and self.latest_map is not None:
            if self.progress_monitor.check_progress(self.latest_map):
                self.get_logger().info(
                    '[Progress Monitor] Current cluster area sufficiently mapped!'
                    '\n- Initiating new frontier detection'
                )
                self.executing = False
                self.goal_start_time = None
                
                # Stop the robot
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                
                # Find new frontier
                self.detect_and_publish_frontier()

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()