import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
import tf_transformations

class PathVisualizer(Node):
    def __init__(self):
        super().__init__('path_visualizer')

        # Subscribe to the planned path
        self.subscription = self.create_subscription(Path, 'planned_path', self.path_callback, 10)

        # Publisher for RViz markers
        self.marker_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)

    def path_callback(self, msg):
        """Receives the Path message and publishes markers for visualization."""
        marker_array = MarkerArray()

        for i, pose in enumerate(msg.poses):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "path_markers"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            marker.pose = pose.pose  # Directly use the pose from Path

            marker.scale.x = 0.3  # Length of arrow
            marker.scale.y = 0.1  # Width of arrow
            marker.scale.z = 0.1  # Thickness of arrow

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Fully visible

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)
        # self.get_logger().info(f'Published {len(marker_array.markers)} markers to RViz')

def main(args=None):
    rclpy.init(args=args)
    node = PathVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
