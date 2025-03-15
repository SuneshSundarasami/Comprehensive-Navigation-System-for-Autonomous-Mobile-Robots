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

    def create_frontier_markers(self, frontier_points, map_info, robot_position, selected_centroid=None, all_centroids=None):
        """Create visualization markers for frontiers and clusters
        
        Args:
            frontier_points: List of frontier points in grid coordinates
            map_info: Map metadata from OccupancyGrid message
            robot_position: Current robot position [x, y]
            selected_centroid: Selected frontier centroid in grid coordinates
            all_centroids: List of all cluster centroids being considered
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

        # Create markers for all frontier points
        if len(frontier_points) > 0:
            points_marker = self._create_points_marker(frontier_points, map_info)
            points_marker.header.frame_id = self.frame_id
            points_marker.header.stamp = stamp
            points_marker.lifetime.sec = 0
            marker_array.markers.append(points_marker)
            
            # Add markers for all considered clusters (red)
            if all_centroids is not None:
                for centroid in all_centroids:
                    if not np.array_equal(centroid, selected_centroid):  # Skip selected centroid
                        cluster_marker = self._create_cluster_marker(centroid, map_info, is_selected=False)
                        cluster_marker.header.frame_id = self.frame_id
                        cluster_marker.header.stamp = stamp
                        cluster_marker.lifetime.sec = 0
                        marker_array.markers.append(cluster_marker)
            
            # Add marker for selected cluster (green)
            if selected_centroid is not None:
                selected_marker = self._create_cluster_marker(selected_centroid, map_info, is_selected=True)
                selected_marker.header.frame_id = self.frame_id
                selected_marker.header.stamp = stamp
                selected_marker.lifetime.sec = 0
                marker_array.markers.append(selected_marker)

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
        
        # Make points smaller and more translucent
        marker.scale.x = 0.05  # Reduced from 0.1
        marker.scale.y = 0.05  # Reduced from 0.1
        marker.color = ColorRGBA(r=0.4, g=0.4, b=1.0, a=0.8)  # Lighter blue, more transparent
        
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

    def _create_cluster_marker(self, centroid, map_info, is_selected=False):
        """Create marker for cluster centroid"""
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = self.marker_id
        self.marker_id += 1
        
        # Smaller size for both selected and unselected
        size = 0.3 if is_selected else 0.2
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = size
        
        # Lighter colors and more translucent
        if is_selected:
            marker.color = ColorRGBA(r=0.2, g=1.0, b=0.2, a=0.7)  # Light green
        else:
            marker.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.4)  # Light red
        
        # Convert to world coordinates
        marker.pose.position.x = centroid[1] * map_info.resolution + map_info.origin.position.x
        marker.pose.position.y = centroid[0] * map_info.resolution + map_info.origin.position.y
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0
        
        return marker