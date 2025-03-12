import rclpy
from rclpy.node import Node
import time
import math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
import tf_transformations
from geometry_msgs.msg import Twist
import numpy as np
import tf2_ros
from geometry_msgs.msg import Pose2D
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy,DurabilityPolicy
from std_srvs.srv import Trigger

class PotentialFieldMappingModel(Node):
    def __init__(self):
        super().__init__('potential_field_node')
        
        # Constants
        self.CONTROL_RATE = 20.0  # Hz
        self.MAX_LINEAR_VEL = 0.1  # m/s
        self.MAX_ANGULAR_VEL = 0.5  # rad/s
        self.SLOW_LINEAR_VEL = 0.05  # m/s near goal
        self.GOAL_POS_TOLERANCE = 0.2  # meters
        self.GOAL_ANGLE_TOLERANCE = 0.1  # radians
        
        # Potential field parameters
        self.ATTRACT_GAIN = 0.7
        self.REPULSE_GAIN = 0.5
        self.OBSTACLE_THRESHOLD = 1.0  # meters
        
        # Initialize state variables
        self.current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.goal_pose = {'x': np.nan, 'y': np.nan, 'theta': np.nan}
        self.v_total = np.zeros(2)
        self.pfield_status = 'Waiting for goal pose'
        
        # Set up transforms
        self.setup_transforms()
        
        # Set up publishers/subscribers with QoS
        self.setup_pub_sub()
        
        # Create control timer
        self.create_timer(1/self.CONTROL_RATE, self.control_loop)

    def setup_transforms(self):
        """Initialize transform listener"""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.transform_matrix = np.eye(4)

    def setup_pub_sub(self):
        """Initialize publishers and subscribers"""
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers
        self.create_subscription(Pose2D, 'end_pose', self.goal_callback, qos_reliable)
        self.create_subscription(Odometry, 'odom', self.odom_callback, qos_reliable)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        
        # Services
        self.create_service(Trigger, 'get_pfield_status', self.get_status_callback)

    def control_loop(self):
        """Main control loop"""
        if np.isnan(self.goal_pose['x']):
            return
            
        # Calculate potential field forces
        v_attract = self.calculate_attractive_force()
        v_repulse = self.calculate_repulsive_force()
        self.v_total = v_attract + v_repulse
        
        # Convert to twist command
        cmd_vel = self.compute_velocity_commands()
        
        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Update status
        self.update_status()

    def calculate_attractive_force(self):
        """Calculate attractive force towards goal"""
        current_pos = np.array([self.current_pose['x'], self.current_pose['y']])
        goal_pos = np.array([self.goal_pose['x'], self.goal_pose['y']])
        
        distance = np.linalg.norm(goal_pos - current_pos)
        if distance < self.GOAL_POS_TOLERANCE:
            return np.zeros(2)
            
        return -self.ATTRACT_GAIN * (current_pos - goal_pos) / distance

    def calculate_repulsive_force(self):
        """Calculate repulsive force from obstacles"""
        v_repulse = np.zeros(2)
        if not hasattr(self, 'obstacle_points'):
            return v_repulse
            
        current_pos = np.array([self.current_pose['x'], self.current_pose['y']])
        
        for obstacle in self.obstacle_points:
            dist = np.linalg.norm(current_pos - obstacle)
            if dist < self.OBSTACLE_THRESHOLD:
                force = self.REPULSE_GAIN * (1/dist - 1/self.OBSTACLE_THRESHOLD) * \
                        (current_pos - obstacle) / (dist**3)
                v_repulse += force
                
        return v_repulse

    def compute_velocity_commands(self):
        """Convert force vector to robot commands"""
        twist = Twist()
        
        # Calculate desired heading
        desired_heading = math.atan2(self.v_total[1], self.v_total[0])
        current_heading = self.current_pose['theta']
        
        # Calculate heading error
        heading_error = math.atan2(math.sin(desired_heading - current_heading),
                                 math.cos(desired_heading - current_heading))
        
        # Set velocities
        distance_to_goal = np.linalg.norm([
            self.current_pose['x'] - self.goal_pose['x'],
            self.current_pose['y'] - self.goal_pose['y']
        ])
        
        max_vel = self.SLOW_LINEAR_VEL if distance_to_goal < 0.5 else self.MAX_LINEAR_VEL
        
        twist.linear.x, twist.angular.z = self.limit_velocities(
            np.linalg.norm(self.v_total),
            heading_error,
            max_vel
        )
        
        return twist

    def update_status(self):
        """Update node status"""
        distance_to_goal = np.linalg.norm([
            self.current_pose['x'] - self.goal_pose['x'],
            self.current_pose['y'] - self.goal_pose['y']
        ])
        
        if distance_to_goal < self.GOAL_POS_TOLERANCE:
            if abs(self.current_pose['theta'] - self.goal_pose['theta']) < self.GOAL_ANGLE_TOLERANCE:
                self.pfield_status = 'Goal reached'
                self.goal_pose = {'x': np.nan, 'y': np.nan, 'theta': np.nan}
            else:
                self.pfield_status = 'Aligning orientation'
        else:
            self.pfield_status = 'Moving to goal'



def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldMappingModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
