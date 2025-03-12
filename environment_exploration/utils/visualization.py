from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
import rclpy
import numpy as np

class FrontierVisualizer:
    def __init__(self):
        self.marker_id = 0
        self.frame_id = "map"  # Use map frame instead of odom
        self.last_markers = {}

    def create_frontier_markers(self, frontier_points, map_info, robot_position, selected_centroid=None):
        """Create visualization markers for frontiers
        
        Args:
            frontier_points: List of frontier points in grid coordinates
            map_info: Map metadata from OccupancyGrid message
            robot_position: Current robot position [x, y]
            selected_centroid: Selected frontier centroid in grid coordinates
        """
        marker_array = MarkerArray()
        
        # Use time 0 to ensure markers are always displayed
        stamp = rclpy.time.Time(seconds=0, nanoseconds=0).to_msg()
        
        # Delete old markers first
        delete_marker = Marker()
        delete_marker.header.frame_id = self.frame_id
        delete_marker.header.stamp = stamp
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # Create new markers with persistent lifetime
        if len(frontier_points) > 0:
            points_marker = self._create_points_marker(frontier_points, map_info)
            points_marker.header.frame_id = self.frame_id
            points_marker.header.stamp = stamp
            points_marker.lifetime.sec = 0  # Make markers persistent
            points_marker.lifetime.nanosec = 0
            marker_array.markers.append(points_marker)
            
            if selected_centroid is not None:
                centroid_marker = self._create_centroid_marker(selected_centroid, map_info)
                centroid_marker.header.frame_id = self.frame_id
                centroid_marker.header.stamp = stamp
                centroid_marker.lifetime.sec = 0
                centroid_marker.lifetime.nanosec = 0
                marker_array.markers.append(centroid_marker)

                path_marker = self._create_path_marker(selected_centroid, map_info, robot_position)
                path_marker.header.frame_id = self.frame_id
                path_marker.header.stamp = stamp
                path_marker.lifetime.sec = 0
                path_marker.lifetime.nanosec = 0
                marker_array.markers.append(path_marker)

        return marker_array

    def _create_delete_marker(self, marker_id, namespace, stamp):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.DELETE
        return marker

    def _create_points_marker(self, frontier_points, map_info):
        marker = Marker()
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.id = self.marker_id
        self.marker_id += 1
        
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
        
        # Convert grid coordinates to world coordinates
        for point in frontier_points:
            p = Point()
            p.x = point[1] * map_info.resolution + map_info.origin.position.x
            p.y = point[0] * map_info.resolution + map_info.origin.position.y
            p.z = 0.05
            marker.points.append(p)
        
        return marker

    def _create_centroid_marker(self, selected_centroid, map_info):
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = self.marker_id
        self.marker_id += 1
        
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        
        marker.pose.position.x = selected_centroid[1] * map_info.resolution + map_info.origin.position.x
        marker.pose.position.y = selected_centroid[0] * map_info.resolution + map_info.origin.position.y
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0
        
        return marker

    def _create_path_marker(self, selected_centroid, map_info, robot_position):
        marker = Marker()
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.id = self.marker_id
        self.marker_id += 1
        
        marker.scale.x = 0.05
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        
        # Start point (robot position)
        start = Point()
        start.x = robot_position[0]
        start.y = robot_position[1]
        start.z = 0.05
        
        # End point (selected frontier)
        end = Point()
        end.x = selected_centroid[1] * map_info.resolution + map_info.origin.position.x
        end.y = selected_centroid[0] * map_info.resolution + map_info.origin.position.y
        end.z = 0.05
        
        marker.points = [start, end]
        return marker