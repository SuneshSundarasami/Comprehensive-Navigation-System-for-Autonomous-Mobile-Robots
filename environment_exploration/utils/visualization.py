from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Time, Duration
import rclpy
import math
import numpy as np

class FrontierVisualizer:
    def __init__(self):
        self.marker_id = 0
        self.frame_id = "map"

    def _set_marker_properties(self, marker, stamp):
        """Set common marker properties"""
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        # Set marker lifetime to persist
        marker.lifetime = Duration(sec=0, nanosec=0)

    def create_frontier_markers(self, frontier_points, map_info, robot_position, selected_centroid=None):
        marker_array = MarkerArray()
        
        # Get current timestamp
        stamp = rclpy.clock.Clock().now().to_msg()
        
        # Delete old markers first
        delete_marker = self._create_delete_marker(stamp)
        marker_array.markers.append(delete_marker)
        
        # Create frontier points marker
        if len(frontier_points) > 0:
            points_marker = self._create_points_marker(frontier_points, map_info, stamp)
            marker_array.markers.append(points_marker)
        
        if selected_centroid is not None:
            # Create selected point marker
            centroid_marker = self._create_centroid_marker(selected_centroid, map_info, stamp)
            marker_array.markers.append(centroid_marker)
            
            # Create path line marker
            path_marker = self._create_path_marker(selected_centroid, map_info, robot_position, stamp)
            marker_array.markers.append(path_marker)
        
        return marker_array

    def _create_delete_marker(self, stamp):
        marker = Marker()
        self._set_marker_properties(marker, stamp)
        marker.ns = "frontier_visualization"
        marker.id = 0
        marker.action = Marker.DELETEALL
        return marker

    def _create_points_marker(self, frontier_points, map_info, stamp):
        marker = Marker()
        self._set_marker_properties(marker, stamp)
        marker.ns = "frontier_points"
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # Set markers scale and color
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)  # Blue

        # Convert frontier points to world coordinates
        for point in frontier_points:
            world_point = Point()
            world_point.x = point[1] * map_info.resolution + map_info.origin.position.x
            world_point.y = point[0] * map_info.resolution + map_info.origin.position.y
            world_point.z = 0.05  # Slightly above ground
            marker.points.append(world_point)

        return marker

    def _create_centroid_marker(self, selected_centroid, map_info, stamp):
        marker = Marker()
        self._set_marker_properties(marker, stamp)
        marker.ns = "selected_centroid"
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Set marker scale and color
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red

        # Set position
        marker.pose.position.x = selected_centroid[1] * map_info.resolution + map_info.origin.position.x
        marker.pose.position.y = selected_centroid[0] * map_info.resolution + map_info.origin.position.y
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0

        return marker

    def _create_path_marker(self, selected_centroid, map_info, robot_position, stamp):
        marker = Marker()
        self._set_marker_properties(marker, stamp)
        marker.ns = "path_to_frontier"
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Set marker scale and color
        marker.scale.x = 0.05  # Line width
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green

        # Add start point (robot position)
        start = Point()
        start.x = robot_position[0]
        start.y = robot_position[1]
        start.z = 0.05
        marker.points.append(start)

        # Add end point (selected frontier)
        end = Point()
        end.x = selected_centroid[1] * map_info.resolution + map_info.origin.position.x
        end.y = selected_centroid[0] * map_info.resolution + map_info.origin.position.y
        end.z = 0.05
        marker.points.append(end)

        return marker

    def clear_markers(self):
        """Create a marker array that clears all previous markers"""
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.frame_id = self.frame_id
        delete_marker.header.stamp = rclpy.clock.Clock().now().to_msg()
        delete_marker.ns = "frontier_visualization"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        return marker_array