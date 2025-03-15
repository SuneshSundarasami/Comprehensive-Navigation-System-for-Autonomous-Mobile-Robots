import rclpy
from rclpy.node import Node
import numpy as np
import tf_transformations
import tf2_ros  
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose2D
from std_srvs.srv import Trigger

class PotentialFieldController(Node):
    def __init__(self):
        super().__init__('potential_field_node')
        
        self.setup_parameters()
        self.setup_transforms()
        self.setup_communications()
        self.initialize_state()
        
        self.create_timer(1.0, self.publish_debug_info)
        self.get_logger().info('Potential field controller initialized')

    def setup_parameters(self):
        # Separate linear and angular velocity limits
        self.linear_vel_max = 0.1
        self.angular_vel_max = 0.5  # Higher angular velocity limit
        self.linear_vel_slow = 0.05
        self.linear_vel_min = 0.01
        
        self.dist_threshold = 0.2  
        self.angle_threshold = 0.1  
        self.slow_zone = 0.5
        
        self.gain_attract = 1.5
        self.gain_repulse = 1
        self.obstacle_radius = 0.8
        self.max_accel = 0.2
        
        # Keep minima detection parameters
        self.stuck_threshold = 0.05
        self.stuck_time_threshold = 3.0
        self.escape_radius = 0.5
        self.escape_angle = np.pi/4
        self.escape_time = 2.0
        self.last_positions = []
        self.last_check_time = None
        self.is_escaping = False
        self.escape_start_time = None

    def setup_transforms(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def setup_communications(self):
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(String, 'pfield_debug', 10)
        
        self.create_subscription(Pose2D, 'end_pose', self.on_goal_received, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom_received, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan_received, 10)
        
        self.status_srv = self.create_service(Trigger, 'get_pfield_status', self.handle_status_request)

    def initialize_state(self):
        self.pose_current = {'pos': np.zeros(2), 'angle': 0.0}
        self.pose_goal = {'pos': np.array([np.nan, np.nan]), 'angle': np.nan}
        self.force_total = np.zeros(2)
        self.force_repulse = np.zeros(2)
        self.cmd_current = Twist()
        self.status = "Waiting for goal pose"

    def on_goal_received(self, msg):
        self.pose_goal = {
            'pos': np.array([msg.x, msg.y]),
            'angle': msg.theta
        }
        self.status = "Moving to goal"
        self.get_logger().info(f"New goal: ({msg.x:.2f}, {msg.y:.2f}, {msg.theta:.2f})")

    def on_odom_received(self, msg):
        if np.isnan(self.pose_goal['pos'][0]):
            return

        self.pose_current['pos'] = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])
        
        _, _, yaw = tf_transformations.euler_from_quaternion([
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        ])
        self.pose_current['angle'] = yaw
        
        if not self.check_goal_reached():
            self.update_control()

    def on_scan_received(self, msg):
        if np.isnan(self.pose_goal['pos'][0]):
            return

        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges[:len(angles)])
        
        valid_idx = ~np.isnan(ranges) & ~np.isinf(ranges)
        angles, ranges = angles[valid_idx], ranges[valid_idx]
        
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        obstacles = np.column_stack((x, y))
        
        self.calculate_repulsion(obstacles)

    def calculate_repulsion(self, obstacles):
        try:
            self.force_repulse = np.zeros(2)
            for obs in obstacles:
                dist = np.linalg.norm(self.pose_current['pos'] - obs)
                if dist < self.obstacle_radius:
                    force = self.gain_repulse * (1/dist - 1/self.obstacle_radius) * \
                           (self.pose_current['pos'] - obs) / (dist**3)
                    self.force_repulse += force
        except Exception as e:
            self.get_logger().error(f'Repulsion calculation error: {str(e)}')

    def check_local_minima(self):
        """Detect if robot is stuck in local minima"""
        current_time = self.get_clock().now()
        
        # Initialize time check
        if self.last_check_time is None:
            self.last_check_time = current_time
            self.last_positions.append(self.pose_current['pos'].copy())
            return False
            
        # Add current position to history
        if (current_time - self.last_check_time).nanoseconds / 1e9 >= 0.5:  # Check every 0.5s
            self.last_check_time = current_time
            self.last_positions.append(self.pose_current['pos'].copy())
            
            # Keep only last 6 positions (3 seconds)
            if len(self.last_positions) > 6:
                self.last_positions.pop(0)
                
            # Check if robot is stuck
            if len(self.last_positions) >= 6:
                max_dist = 0
                for pos in self.last_positions:
                    dist = np.linalg.norm(pos - self.last_positions[-1])
                    max_dist = max(max_dist, dist)
                
                if max_dist < self.stuck_threshold:
                    self.get_logger().warn("Local minima detected!")
                    return True
        
        return False

    def escape_local_minima(self):
        """Generate escape behavior from local minima"""
        current_time = self.get_clock().now()
        
        # Start escape behavior
        if not self.is_escaping:
            self.is_escaping = True
            self.escape_start_time = current_time
            self.last_positions.clear()
            self.get_logger().info("Starting escape maneuver")
            
            # Calculate escape direction (perpendicular to goal direction)
            to_goal = self.pose_goal['pos'] - self.pose_current['pos']
            escape_angle = np.arctan2(to_goal[1], to_goal[0]) + self.escape_angle
            self.escape_direction = np.array([
                np.cos(escape_angle),
                np.sin(escape_angle)
            ])
            
        # Check if escape time is over
        if (current_time - self.escape_start_time).nanoseconds / 1e9 >= self.escape_time:
            self.is_escaping = False
            self.escape_start_time = None
            self.get_logger().info("Escape maneuver completed")
            return None
            
        # Generate escape command
        cmd = Twist()
        cmd.linear.x = self.linear_vel_max * 0.5  # Half speed during escape
        
        # Adjust heading to escape direction
        heading_error = np.arctan2(self.escape_direction[1], self.escape_direction[0]) - self.pose_current['angle']
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        cmd.angular.z = np.clip(heading_error * 2.0, -self.angular_vel_max, self.angular_vel_max)
        
        return cmd

    def update_control(self):
        """Main control loop with state machine logic"""
        try:
            cmd = Twist()
            
            # If in Allignment phase, only handle rotation
            if self.status == "Alligning orientation":
                angle_error = self.pose_goal['angle'] - self.pose_current['angle']
                angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
                
                cmd.linear.x = 0.0  # Pure rotation
                cmd.angular.z = np.clip(angle_error * 2.0, -self.angular_vel_max, self.angular_vel_max)
                
                self.cmd_vel_pub.publish(cmd)
                self.cmd_current = cmd
                return
                
            # Regular motion control
            to_goal = self.pose_goal['pos'] - self.pose_current['pos']
            dist = np.linalg.norm(to_goal)
            
            # Calculate direct heading to goal
            goal_heading = np.arctan2(to_goal[1], to_goal[0])
            heading_error = goal_heading - self.pose_current['angle']
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            # Calculate forces
            force_attract = self.gain_attract * to_goal / dist
            self.force_total = force_attract + self.force_repulse
            
            # Check if path is clear
            path_is_clear = np.linalg.norm(self.force_repulse) < 0.1
            
            # Check for local minima and execute escape if needed
            if not self.is_escaping and self.check_local_minima():
                escape_cmd = self.escape_local_minima()
                if escape_cmd:
                    self.cmd_vel_pub.publish(escape_cmd)
                    self.cmd_current = escape_cmd
                    return
            
            # Forward motion control
            if abs(heading_error) > np.pi/3:  # Large heading error
                # Pure rotation for large heading errors
                cmd.linear.x = 0.0
                cmd.angular.z = np.clip(heading_error * 2.0, -self.angular_vel_max, self.angular_vel_max)
            elif abs(heading_error) < 0.1 and path_is_clear:
                # Direct forward motion when well-aligned
                cmd.linear.x = self.linear_vel_max
                if dist < self.slow_zone:
                    # Modified velocity scaling for smoother approach
                    progress = dist / self.slow_zone
                    cmd.linear.x = max(
                        self.linear_vel_min,
                        self.linear_vel_max * progress
                    )
                cmd.angular.z = np.clip(heading_error * 2.0, -self.angular_vel_max, self.angular_vel_max)
            else:
                # Normal potential field motion
                vel_mag = np.linalg.norm(self.force_total)
                vel_ang = np.arctan2(self.force_total[1], self.force_total[0])
                
                # Maintain higher minimum velocity
                max_vel = self.linear_vel_max
                if dist < self.slow_zone:
                    progress = dist / self.slow_zone
                    max_vel = max(
                        self.linear_vel_min,
                        self.linear_vel_max * progress
                    )
                
                linear, angular = self.scale_velocities(vel_mag, heading_error, max_vel)
                cmd.linear.x = linear
                cmd.angular.z = angular
            
            self.cmd_vel_pub.publish(cmd)
            self.cmd_current = cmd
                
        except Exception as e:
            self.get_logger().error(f'Control update error: {str(e)}')
            self.stop_robot()

    def scale_velocities(self, linear, angular, max_vel):
        """Scale velocities with separate limits"""
        if abs(angular) > np.pi/4:
            return max_vel * 0.2, np.clip(angular, -self.angular_vel_max, self.angular_vel_max)
        
        # Scale linear and angular velocities separately
        linear_scale = self.linear_vel_max / max(abs(linear), self.linear_vel_max)
        angular_scale = self.angular_vel_max / max(abs(angular), self.angular_vel_max)
        
        return linear * linear_scale, angular * angular_scale

    def smooth_velocity(self, target_vel):
        """Smooth velocity transitions including reverse motion"""
        current_vel = self.cmd_current.linear.x
        vel_diff = target_vel - current_vel
        vel_change = np.clip(vel_diff, -self.max_accel, self.max_accel)
        return current_vel + vel_change

    def check_goal_reached(self):
        dist = np.linalg.norm(self.pose_current['pos'] - self.pose_goal['pos'])
        angle_error = self.pose_goal['angle'] - self.pose_current['angle']
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        # Increase distance threshold slightly
        if dist < self.dist_threshold:
            if abs(angle_error) < self.angle_threshold:
                self.status = "Goal Position Reached! Alligned orientation!"  
                self.stop_robot()
                return True
            else:
                if self.status != "Alligning orientation":
                    self.get_logger().info(
                        f"Starting orientation Allignment. Error: {np.degrees(angle_error):.1f}°"
                    )
                self.status = "Alligning orientation"
                return False
        
        if self.status != "Moving to goal":
            self.get_logger().info(f"Moving to goal. Distance: {dist:.2f}m")
        self.status = "Moving to goal"
        return False

    def handle_status_request(self, request, response):
        response.success = True
        response.message = self.status
        return response

    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def publish_debug_info(self):
        try:
            if not np.isnan(self.pose_goal['pos'][0]):
                dist = np.linalg.norm(self.pose_goal['pos'] - self.pose_current['pos'])
                angle_error = self.pose_goal['angle'] - self.pose_current['angle']
                angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
                
                debug_msg = String()
                debug_msg.data = (
                    f"Status: {self.status}\n"
                    f"Position: [{self.pose_current['pos'][0]:.2f}, {self.pose_current['pos'][1]:.2f}]\n"
                    f"Goal: [{self.pose_goal['pos'][0]:.2f}, {self.pose_goal['pos'][1]:.2f}]\n"
                    f"Distance: {dist:.2f}m\n"
                    f"Angle error: {np.degrees(angle_error):.1f}°\n"
                    f"Velocities - Linear: {self.cmd_current.linear.x:.2f} m/s, "
                    f"Angular: {self.cmd_current.angular.z:.2f} rad/s\n"
                    f"Total force: [{self.force_total[0]:.2f}, {self.force_total[1]:.2f}]"
                )
                self.debug_pub.publish(debug_msg)
                self.get_logger().info(debug_msg.data)
                
        except Exception as e:
            self.get_logger().error(f"Debug info error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
