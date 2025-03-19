import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np

class TFRepublisher(Node):
    def __init__(self):
        super().__init__('tf_republisher')
        
        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Latest transform storage
        self.latest_transform = None
        
        # Create timer for republishing (50Hz)
        self.create_timer(0.02, self.republish_transform)
        
        self.get_logger().info('TF Republisher node initialized')

    def republish_transform(self):
        try:
            # Look up the latest transform
            transform = self.tf_buffer.lookup_transform(
                'map',
                'odom',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            
            # Store and republish with current timestamp
            self.latest_transform = transform
            
            # Update timestamp to current time
            transform.header.stamp = self.get_clock().now().to_msg()
            
            # Republish transform
            self.tf_broadcaster.sendTransform(transform)
            
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            # If we have a stored transform, republish it
            if self.latest_transform is not None:
                self.latest_transform.header.stamp = self.get_clock().now().to_msg()
                self.tf_broadcaster.sendTransform(self.latest_transform)
            else:
                if not hasattr(self, 'warning_throttle'):
                    self.warning_throttle = self.get_clock().now()
                # Throttle warnings to once per second
                current_time = self.get_clock().now()
                if (current_time - self.warning_throttle).nanoseconds / 1e9 >= 1.0:
                    self.get_logger().warning(f'Transform lookup failed: {str(e)}')
                    self.warning_throttle = current_time

def main(args=None):
    rclpy.init(args=args)
    node = TFRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()