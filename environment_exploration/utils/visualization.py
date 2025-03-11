from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import math

class FrontierVisualizer:
    def __init__(self):
        self.marker_id = 0

    def create_frontier_markers(self, frontier_points, map_info, robot_position, selected_centroid=None):
        marker_array = MarkerArray()
        
        # Create frontier points marker
        points_marker = self._create_points_marker(frontier_points, map_info)
        marker_array.markers.append(points_marker)
        
        if selected_centroid is not None:
            # Create arrow marker
            arrow_marker = self._create_arrow_marker(selected_centroid, map_info, robot_position)
            marker_array.markers.append(arrow_marker)
            
            # Create sphere marker
            sphere_marker = self._create_sphere_marker(selected_centroid, map_info)
            marker_array.markers.append(sphere_marker)
        
        return marker_array

    def _create_points_marker(self, frontier_points, map_info):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "frontier_points"
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)

        for point in frontier_points:
            p = Point()
            p.x = point[1] * map_info.resolution + map_info.origin.position.x
            p.y = point[0] * map_info.resolution + map_info.origin.position.y
            p.z = 0.0
            marker.points.append(p)

        return marker

    def _create_arrow_marker(self, selected_centroid, map_info, robot_position):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "selected_frontier"
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)

        marker.pose.position.x = selected_centroid[1] * map_info.resolution + map_info.origin.position.x
        marker.pose.position.y = selected_centroid[0] * map_info.resolution + map_info.origin.position.y
        marker.pose.position.z = 0.2

        dx = marker.pose.position.x - robot_position[0]
        dy = marker.pose.position.y - robot_position[1]
        yaw = math.atan2(dy, dx)
        marker.pose.orientation.z = math.sin(yaw / 2.0)
        marker.pose.orientation.w = math.cos(yaw / 2.0)

        return marker

    def _create_sphere_marker(self, selected_centroid, map_info):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "selected_frontier_sphere"
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)

        marker.pose.position.x = selected_centroid[1] * map_info.resolution + map_info.origin.position.x
        marker.pose.position.y = selected_centroid[0] * map_info.resolution + map_info.origin.position.y
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0

        return marker