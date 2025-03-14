import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
from localization.motion_model import MotionModel
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import time

class MotionModelTester(Node):
    def __init__(self):
        super().__init__('motion_model_tester')
        
        # Initialize motion model
        self.motion_model = MotionModel()
        
        # Create test particles
        self.particles = np.array([
            [-0.913, -4.88, 0.0],      # First particle at initial pose
            [-0.413, -4.38, np.pi/4],  # Second particle offset and rotated
            [-1.413, -5.38, -np.pi/4]  # Third particle differently offset
        ])
        
        # Subscribe to cmd_vel
        self.cmd_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Publisher for visualization
        self.marker_pub = self.create_publisher(MarkerArray, 'particle_markers', 10)
        
        # Timer for visualization updates
        self.timer = self.create_timer(0.1, self.visualize_particles)
        
        self.last_update_time = self.get_clock().now()
        self.get_logger().info('Motion Model Tester initialized')

    def cmd_vel_callback(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = current_time
        
        # Debug incoming cmd_vel with scaled velocities
        scaled_v = msg.linear.x * dt
        scaled_w = msg.angular.z * dt
        
        self.get_logger().info(
            f'\nReceived cmd_vel:'
            f'\n - linear.x: {msg.linear.x:.3f} m/s'
            f'\n - angular.z: {msg.angular.z:.3f} rad/s'
            f'\n - dt: {dt:.3f} s'
            f'\n - scaled movement: {scaled_v:.3f} m'
            f'\n - scaled rotation: {scaled_w:.3f} rad'
        )
        
        if dt < 0.01:
            self.get_logger().warn(f'Skipping update - dt {dt:.3f} too small')
            return
        
        # Store initial positions for debugging
        initial_poses = self.particles.copy()
        
        # Scale velocities by a factor to reduce magnitude
        scale_factor = 0.1  # Adjust this value to tune movement magnitude
        
        # Update particles using motion model with scaled velocities
        self.particles = self.motion_model.update_particles(
            self.particles,
            msg.linear.x * scale_factor,
            msg.angular.z * scale_factor,
            dt
        )
        
        # Calculate and log movement for each particle
        for i, (init_pose, new_pose) in enumerate(zip(initial_poses, self.particles)):
            dx = new_pose[0] - init_pose[0]
            dy = new_pose[1] - init_pose[1]
            dtheta = new_pose[2] - init_pose[2]
            
            # Normalize angle difference to [-pi, pi]
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
            
            movement_distance = np.sqrt(dx*dx + dy*dy)
            expected_distance = abs(msg.linear.x * dt * scale_factor)
            
            self.get_logger().info(
                f'\nParticle {i} update:'
                f'\n - Initial: ({init_pose[0]:.3f}, {init_pose[1]:.3f}, {init_pose[2]:.3f})'
                f'\n - Final:   ({new_pose[0]:.3f}, {new_pose[1]:.3f}, {new_pose[2]:.3f})'
                f'\n - Delta:   dx={dx:.3f}, dy={dy:.3f}, dθ={dtheta:.3f}'
                f'\n - Movement distance: {movement_distance:.3f}m'
                f'\n - Expected distance: {expected_distance:.3f}m'
            )

    def visualize_particles(self):
        marker_array = MarkerArray()
        
        for i, particle in enumerate(self.particles):
            # Create arrow marker for each particle
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "particles"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Set position
            marker.pose.position.x = particle[0]
            marker.pose.position.y = particle[1]
            marker.pose.position.z = 0.0
            
            # Set orientation (convert theta to quaternion)
            marker.pose.orientation.z = np.sin(particle[2] / 2.0)
            marker.pose.orientation.w = np.cos(particle[2] / 2.0)
            
            # Set scale
            marker.scale.x = 0.3  # Arrow length
            marker.scale.y = 0.05  # Arrow width
            marker.scale.z = 0.05  # Arrow height
            
            # Set color
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

def main():
    rclpy.init()
    node = MotionModelTester()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()