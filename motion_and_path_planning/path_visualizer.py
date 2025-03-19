import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
import tf_transformations

class PathVisualizer(Node):
    def __init__(self):
        super().__init__('path_visualizer')
        self.subscription = self.create_subscription(Path, 'planned_path', self.path_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        self.get_logger().info('Path visualizer initialized')

    def path_callback(self, msg):
        """Receives the Path message and publishes markers for visualization."""
        marker_array = MarkerArray()

        # First, add a deletion marker to clear previous markers
        delete_marker = Marker()
        delete_marker.header.frame_id = "map"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = "path_visualization"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # Then add new markers for the current path
        for i, pose in enumerate(msg.poses):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "path_visualization"
            marker.id = i + 1
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            marker.pose = pose.pose

            # Smaller arrow sizes
            marker.scale.x = 0.15  # Length of arrow (was 0.5)
            marker.scale.y = 0.05  # Width of arrow (was 0.15)
            marker.scale.z = 0.05  # Thickness of arrow (was 0.15)

            # Color gradient from red to green with better visibility
            progress = float(i) / max(len(msg.poses) - 1, 1)
            marker.color.r = 1.0 - progress  # Start red, fade out
            marker.color.g = progress  # Fade in green
            marker.color.b = 0.0
            marker.color.a = 0.8  # Slightly transparent for better visibility

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)
        self.get_logger().debug(
            f'Published path visualization:'
            f'\n- Total markers: {len(msg.poses)}'
            f'\n- Arrow size: 0.15m x 0.05m'
        )

def main(args=None):
    rclpy.init(args=args)
    node = PathVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
