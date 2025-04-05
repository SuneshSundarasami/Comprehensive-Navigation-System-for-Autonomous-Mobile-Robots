import rclpy
import tf2_ros
import numpy as np
import tf_transformations
from geometry_msgs.msg import TransformStamped

def get_latest_transform(tf_buffer, target_frame, source_frame, node):
    """Get latest available transform with fallback to transform chain."""
    try:
        # Try direct transform with latest available
        transform = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),  # Use latest available transform
            timeout=rclpy.duration.Duration(seconds=0.1)
        )
        return transform
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
            tf2_ros.ExtrapolationException) as e:
        try:
            # Fallback: Try transform chain through odom
            odom_to_source = tf_buffer.lookup_transform(
                'odom',
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            target_to_odom = tf_buffer.lookup_transform(
                target_frame,
                'odom',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Combine transforms
            combined = TransformStamped()
            combined.header.frame_id = target_frame
            combined.child_frame_id = source_frame
            combined.header.stamp = node.get_clock().now().to_msg()
            
            # Combine translations
            combined.transform.translation.x = (
                target_to_odom.transform.translation.x + 
                odom_to_source.transform.translation.x
            )
            combined.transform.translation.y = (
                target_to_odom.transform.translation.y + 
                odom_to_source.transform.translation.y
            )
            
            # Combine rotations
            q1 = [
                target_to_odom.transform.rotation.x,
                target_to_odom.transform.rotation.y,
                target_to_odom.transform.rotation.z,
                target_to_odom.transform.rotation.w
            ]
            q2 = [
                odom_to_source.transform.rotation.x,
                odom_to_source.transform.rotation.y,
                odom_to_source.transform.rotation.z,
                odom_to_source.transform.rotation.w
            ]
            combined_q = tf_transformations.quaternion_multiply(q1, q2)
            combined.transform.rotation.x = combined_q[0]
            combined.transform.rotation.y = combined_q[1]
            combined.transform.rotation.z = combined_q[2]
            combined.transform.rotation.w = combined_q[3]
            
            return combined
            
        except Exception as chain_error:
            raise Exception(f"Transform error: {str(e)}\nChain error: {str(chain_error)}")