#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped
import tf_transformations
import numpy as np

class TFRepublisher(Node):
    def __init__(self):
        super().__init__('tf_republisher')
        
        # Create TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Create pose publisher
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            'robot_position',
            10
        )
        
        # Timer for publishing position
        self.create_timer(0.1, self.publish_robot_position)  # 10Hz
        
        self.get_logger().info('TF Republisher initialized')

    def publish_robot_position(self):
        """Get position using transform chain and publish"""
        try:
            # Get transform components
            odom_to_base = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            map_to_odom = self.tf_buffer.lookup_transform(
                'map',
                'odom',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Create PoseStamped message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'
            
            # Combine translations
            pose_msg.pose.position.x = (
                map_to_odom.transform.translation.x + 
                odom_to_base.transform.translation.x
            )
            pose_msg.pose.position.y = (
                map_to_odom.transform.translation.y + 
                odom_to_base.transform.translation.y
            )
            pose_msg.pose.position.z = 0.0
            
            # Combine rotations
            q1 = [
                map_to_odom.transform.rotation.x,
                map_to_odom.transform.rotation.y,
                map_to_odom.transform.rotation.z,
                map_to_odom.transform.rotation.w
            ]
            q2 = [
                odom_to_base.transform.rotation.x,
                odom_to_base.transform.rotation.y,
                odom_to_base.transform.rotation.z,
                odom_to_base.transform.rotation.w
            ]
            combined_q = tf_transformations.quaternion_multiply(q1, q2)
            
            pose_msg.pose.orientation.x = combined_q[0]
            pose_msg.pose.orientation.y = combined_q[1]
            pose_msg.pose.orientation.z = combined_q[2]
            pose_msg.pose.orientation.w = combined_q[3]
            
            # Publish the pose
            self.pose_publisher.publish(pose_msg)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            if not hasattr(self, '_last_tf_error_time') or \
               (self.get_clock().now() - self._last_tf_error_time).nanoseconds / 1e9 > 5.0:
                self.get_logger().warn(f'Transform error: {str(e)}')
                self._last_tf_error_time = self.get_clock().now()

def main(args=None):
    rclpy.init(args=args)
    node = TFRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()