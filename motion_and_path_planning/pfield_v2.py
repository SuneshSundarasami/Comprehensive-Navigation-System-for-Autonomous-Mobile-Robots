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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_srvs.srv import Trigger

class PotentialFieldMappingModel(Node):
    def __init__(self):
        super().__init__('potential_field_node')
        
        # Constants
        self.CONTROL_RATE = 20.0  # Hz
        self.MAX_VEL = 0.25  # m/s
        self.SLOW_VEL = 0.15  # Increased from 0.1
        self.MIN_VEL = 0.05  # Added minimum velocity
        self.GOAL_THRESHOLD = 0.5  # meters
        self.HEADING_THRESHOLD = 0.1  # radians
        self.SLOW_ZONE = 0.8  # Distance to start slowing (meters)
        
        # Potential field parameters
        self.ATTRACT_GAIN = 1.0
        self.REPULSE_GAIN = 0.5  # Increased for better obstacle avoidance
        self.OBSTACLE_THRESHOLD = 1.0  # meters
        
        # Add debug publisher
        self.debug_pub = self.create_publisher(String, 'pfield_debug', 10)
        
        self._setup_transforms()
        self._setup_publishers_subscribers()
        self._initialize_state()
        
        # Add timer for regular status updates
        self.create_timer(1.0, self._debug_callback)
        self.get_logger().info('Potential Field Node initialized')

    def _setup_transforms(self):
        """Initialize transform handling"""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def _setup_publishers_subscribers(self):
        """Set up ROS communications"""
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.create_subscription(Pose2D, 'end_pose', self._goal_callback, 10)
        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_callback, 10)
        
        # Service
        self.status_service = self.create_service(
            Trigger, 'get_pfield_status', self._status_callback)

    def _initialize_state(self):
        """Initialize state variables"""
        self.current_pose = {'position': np.zeros(2), 'orientation': 0.0}
        self.goal_pose = {'position': np.array([np.nan, np.nan]), 'orientation': np.nan}
        self.v_total = np.zeros(2)
        self.status = "Waiting for goal"

    def _goal_callback(self, msg):
        """Handle new goal"""
        self.goal_pose = {
            'position': np.array([msg.x, msg.y]),
            'orientation': msg.theta
        }
        self.status = "Moving to goal"
        self.get_logger().info(f"Received new goal: ({msg.x}, {msg.y}, {msg.theta})")

    def _odom_callback(self, msg):
        """Process odometry updates"""
        if np.isnan(self.goal_pose['position'][0]):
            return

        # Update current pose
        self.current_pose['position'] = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])
        
        # Get yaw from quaternion
        _, _, yaw = tf_transformations.euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        self.current_pose['orientation'] = yaw
        
        # Calculate control if not at goal
        if not self._at_goal():
            self._calculate_and_publish_control()

    def _scan_callback(self, msg):
        """Process laser scan data"""
        if np.isnan(self.goal_pose['position'][0]):
            return

        # Convert scan to cartesian coordinates
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges[:len(angles)])
        
        # Filter out invalid readings
        valid_idx = ~np.isnan(ranges) & ~np.isinf(ranges)
        angles, ranges = angles[valid_idx], ranges[valid_idx]
        
        # Convert to cartesian
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        obstacles = np.column_stack((x, y))
        
        # Calculate repulsive forces
        self._calculate_repulsive_forces(obstacles)

    def _calculate_repulsive_forces(self, obstacles):
        """Calculate repulsive forces from obstacles"""
        try:
            self.v_repulse = np.zeros(2)
            for obstacle in obstacles:
                dist = np.linalg.norm(self.current_pose['position'] - obstacle)
                if dist < self.OBSTACLE_THRESHOLD:
                    force = self.REPULSE_GAIN * (1/dist - 1/self.OBSTACLE_THRESHOLD) * \
                            (self.current_pose['position'] - obstacle) / (dist**3)
                    self.v_repulse += force
        except Exception as e:
            self.get_logger().error(f'Error calculating repulsive forces: {str(e)}')

    def _calculate_and_publish_control(self):
        """Calculate and publish control commands"""
        try:
            # Calculate attractive force
            to_goal = self.goal_pose['position'] - self.current_pose['position']
            dist_to_goal = np.linalg.norm(to_goal)
            v_attract = self.ATTRACT_GAIN * to_goal / dist_to_goal
            
            # Get latest obstacle forces
            v_repulse = self.v_repulse if hasattr(self, 'v_repulse') else np.zeros(2)
            
            # Combine forces
            self.v_total = v_attract + v_repulse
            
            # Create control command
            cmd = Twist()
            v_mag = np.linalg.norm(self.v_total)
            v_ang = np.arctan2(self.v_total[1], self.v_total[0])
            
            # Progressive velocity scaling based on distance
            if dist_to_goal < self.SLOW_ZONE:
                # Linear interpolation between MIN_VEL and SLOW_VEL
                progress = dist_to_goal / self.SLOW_ZONE
                max_vel = self.MIN_VEL + (self.SLOW_VEL - self.MIN_VEL) * progress
            else:
                max_vel = self.MAX_VEL
            
            # Calculate heading error
            heading_error = v_ang - self.current_pose['orientation']
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            # Set velocities with smoother transitions
            linear, angular = self._limit_velocities(v_mag, heading_error, max_vel)
            cmd.linear.x = self._smooth_velocity(linear)
            cmd.angular.z = angular
            
            self.cmd_vel_pub.publish(cmd)
            self.current_cmd = cmd
            
        except Exception as e:
            self.get_logger().error(f'Error in control calculation: {str(e)}')
            self._stop_robot()

    def _smooth_velocity(self, target_vel):
        """Apply smoothing to velocity changes"""
        current_vel = getattr(getattr(self, 'current_cmd', Twist()).linear, 'x', 0.0)
        max_accel = 0.1  # Maximum acceleration per control cycle
        
        # Limit acceleration
        vel_diff = target_vel - current_vel
        vel_change = np.clip(vel_diff, -max_accel, max_accel)
        return current_vel + vel_change

    def _limit_velocities(self, linear, angular, max_vel):
        """Scale velocities to respect limits"""
        if abs(angular) > np.pi/4:
            # Allow some forward motion during rotation
            return max_vel * 0.2, np.clip(angular, -max_vel, max_vel)
        
        # Scale both velocities proportionally if needed
        scale = max_vel / max(abs(linear), abs(angular), max_vel)
        return linear * scale, angular * scale

    def _at_goal(self):
        """Check if robot has reached the goal"""
        if np.linalg.norm(self.current_pose['position'] - self.goal_pose['position']) < self.GOAL_THRESHOLD:
            if abs(self.current_pose['orientation'] - self.goal_pose['orientation']) < self.HEADING_THRESHOLD:
                self.status = "Goal reached"
                return True
            self.status = "Aligning orientation"
        return False

    def _status_callback(self, request, response):
        """Handle status service requests"""
        response.success = True
        response.message = self.status
        return response

    def _stop_robot(self):
        """Emergency stop function"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def _debug_callback(self):
        """Publish debug information"""
        try:
            # Calculate distance to goal if goal exists
            if not np.isnan(self.goal_pose['position'][0]):
                dist_to_goal = f"{np.linalg.norm(self.goal_pose['position'] - self.current_pose['position']):.2f}m"
            else:
                dist_to_goal = "N/A"

            # Get current velocities
            current_vel = getattr(self, 'current_cmd', Twist())
            linear_vel = getattr(current_vel.linear, 'x', 0.0)
            angular_vel = getattr(current_vel.angular, 'z', 0.0)

            debug_msg = String()
            debug_msg.data = (
                f"Status: {self.status}\n"
                f"Current pos: [{self.current_pose['position'][0]:.2f}, {self.current_pose['position'][1]:.2f}]\n"
                f"Goal pos: [{self.goal_pose['position'][0]:.2f}, {self.goal_pose['position'][1]:.2f}]\n"
                f"Distance to goal: {dist_to_goal}\n"
                f"Velocities - Linear: {linear_vel:.2f} m/s, Angular: {angular_vel:.2f} rad/s\n"
                f"Total force: [{self.v_total[0]:.2f}, {self.v_total[1]:.2f}]"
            )
            self.debug_pub.publish(debug_msg)
            self.get_logger().info(debug_msg.data)
        except Exception as e:
            self.get_logger().error(f"Error in debug callback: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldMappingModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
