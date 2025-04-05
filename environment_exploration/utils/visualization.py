from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
import rclpy
import numpy as np

class FrontierVisualizer:
    def __init__(self):
        self.marker_id = 0
        self.frame_id = "map"
        self.last_markers = {}

    def create_frontier_markers(self, frontier_points, map_info, robot_position, selected_centroid=None, all_centroids=None, cluster_labels=None):
        """Create visualization markers for frontiers and clusters"""
        marker_array = MarkerArray()
        stamp = rclpy.time.Time(seconds=0, nanoseconds=0).to_msg()
        
        # Delete old markers
        delete_marker = Marker()
        delete_marker.header.frame_id = self.frame_id
        delete_marker.header.stamp = stamp
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        if len(frontier_points) > 0 and cluster_labels is not None:
            # Create markers for unselected cluster points (red)
            unselected_points = []
            selected_points = []
            
            # Sort points into selected and unselected based on cluster
            for point, label in zip(frontier_points, cluster_labels):
                if selected_centroid is not None:
                    # Check if point belongs to selected cluster
                    if np.array_equal(np.mean(frontier_points[cluster_labels == label], axis=0), selected_centroid):
                        selected_points.append(point)
                    else:
                        unselected_points.append(point)
                else:
                    unselected_points.append(point)
            
            # Create marker for unselected points
            if unselected_points:
                unselected_marker = self._create_points_marker(
                    unselected_points, 
                    map_info,
                    ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)  # Red
                )
                unselected_marker.header.frame_id = self.frame_id
                unselected_marker.header.stamp = stamp
                unselected_marker.lifetime.sec = 0
                unselected_marker.ns = "unselected_points"
                marker_array.markers.append(unselected_marker)
            
            # Create marker for selected points
            if selected_points:
                selected_marker = self._create_points_marker(
                    selected_points, 
                    map_info,
                    ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9)  # Green
                )
                selected_marker.header.frame_id = self.frame_id
                selected_marker.header.stamp = stamp
                selected_marker.lifetime.sec = 0
                selected_marker.ns = "selected_points"
                marker_array.markers.append(selected_marker)
            
            # Add cross marker for selected centroid
            if selected_centroid is not None:
                cross_marker = self._create_cross_marker(selected_centroid, map_info)
                cross_marker.header.frame_id = self.frame_id
                cross_marker.header.stamp = stamp
                cross_marker.lifetime.sec = 0
                cross_marker.ns = "centroid_cross"
                marker_array.markers.append(cross_marker)
        
        return marker_array

    def _create_delete_marker(self, marker_id, namespace, stamp):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.DELETE
        return marker

    def _create_points_marker(self, points, map_info, color):
        """Create marker for points with specified color"""
        marker = Marker()
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.id = self.marker_id
        self.marker_id += 1
        
        # Point size
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.color = color
        
        # Convert grid coordinates to world coordinates
        for point in points:
            p = Point()
            p.x = point[1] * map_info.resolution + map_info.origin.position.x
            p.y = point[0] * map_info.resolution + map_info.origin.position.y
            p.z = 0.1
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



    def _create_cross_marker(self, centroid, map_info, size=0.2, thickness=0.02):
        """Create an X-shaped cross marker at the centroid"""
        marker = Marker()
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.id = self.marker_id
        self.marker_id += 1
        
        # Cross size and color
        marker.scale.x = thickness  # Line thickness
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Bright red
        
        # Convert centroid to world coordinates
        x = centroid[1] * map_info.resolution + map_info.origin.position.x
        y = centroid[0] * map_info.resolution + map_info.origin.position.y
        z = 0.15  # Height above ground
        
        # Create X-shaped cross points (two diagonal lines)
        p1 = Point(x=x-size/2, y=y-size/2, z=z)  # Bottom-left
        p2 = Point(x=x+size/2, y=y+size/2, z=z)  # Top-right
        p3 = Point(x=x-size/2, y=y+size/2, z=z)  # Top-left
        p4 = Point(x=x+size/2, y=y-size/2, z=z)  # Bottom-right
        
        # Add diagonal lines
        marker.points.extend([p1, p2])  # First diagonal
        marker.points.extend([p3, p4])  # Second diagonal
        
        return marker