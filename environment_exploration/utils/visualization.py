from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import rclpy
import numpy as np

class FrontierVisualizer:
    def __init__(self,logger=None):
        self.marker_id = 0
        self.frame_id = "map"
        self.logger = logger


    def create_frontier_markers(self, frontier_points, map_info, robot_position, 
                              selected_point=None, all_centroids=None, 
                              cluster_labels=None):
        """Create visualization markers for frontiers"""
        marker_array = MarkerArray()
        selected_cluster = -1  # Initialize selected_cluster
        
        # Debug logging
        if self.logger:
            self.logger.info(
                f'Creating frontier markers:'
                f'\n- Frontier points: {len(frontier_points)}'
                f'\n- Has clusters: {cluster_labels is not None}'
                f'\n- Has selected point: {selected_point is not None}'
            )

        # Create delete marker
        delete_marker = Marker()
        delete_marker.header.frame_id = self.frame_id
        delete_marker.header.stamp = rclpy.time.Time().to_msg()
        delete_marker.action = Marker.DELETEALL
        delete_marker.id = 0
        marker_array.markers.append(delete_marker)

        if len(frontier_points) == 0:
            if self.logger:
                self.logger.warn('No frontier points to visualize')
            return marker_array

        # Handle clustering
        if cluster_labels is not None:
            unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
            
            if self.logger:
                self.logger.info(f'Processing {len(unique_clusters)} clusters')

            # Single cluster case
            if len(unique_clusters) == 1:
                selected_cluster = unique_clusters[0]
                if self.logger:
                    self.logger.info('Single cluster detected - marking as selected')

            # Multiple clusters case
            elif selected_point is not None:
                selected_point_idx = np.where(np.all(frontier_points == selected_point, axis=1))[0]
                if len(selected_point_idx) > 0:
                    selected_cluster = cluster_labels[selected_point_idx[0]]
                    if self.logger:
                        self.logger.info(f'Selected cluster identified: {selected_cluster}')

            # Create markers for each cluster
            for i, cluster_id in enumerate(unique_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_points = frontier_points[cluster_mask]
                
                if len(cluster_points) > 0:
                    if cluster_id == selected_cluster:
                        color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9)  # Green
                        if self.logger:
                            self.logger.info(f'Marking cluster {cluster_id} as selected (green)')
                    else:
                        color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)  # Red
                        if self.logger:
                            self.logger.debug(f'Marking cluster {cluster_id} as unselected (red)')
                    
                    marker = self._create_points_marker(cluster_points, map_info, color)
                    marker.header.frame_id = self.frame_id
                    marker.header.stamp = rclpy.time.Time().to_msg()
                    marker.id = i + 1
                    marker.ns = f"cluster_{cluster_id}"
                    marker_array.markers.append(marker)

        # Add selected point marker
        if selected_point is not None:
            cross = self._create_cross_marker(selected_point, map_info)
            cross.header.frame_id = self.frame_id
            cross.header.stamp = rclpy.time.Time().to_msg()
            cross.id = len(marker_array.markers)
            cross.ns = "selected_point"
            marker_array.markers.append(cross)
            if self.logger:
                self.logger.info(f'Added cross marker at point {selected_point}')

        # Final logging
        if self.logger:
            self.logger.info(
                f'Visualization complete:'
                f'\n- Created {len(marker_array.markers)} markers'
                f'\n- Selected cluster: {selected_cluster}'
                f'\n- Cross marker: {selected_point is not None}'
            )

        return marker_array

    def _create_points_marker(self, points, map_info, color):
        """Create marker for points"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color = color
        
        for point in points:
            p = Point()
            p.x = point[1] * map_info.resolution + map_info.origin.position.x
            p.y = point[0] * map_info.resolution + map_info.origin.position.y
            p.z = 0.1
            marker.points.append(p)
        
        return marker

    def _create_cross_marker(self, point, map_info, size=0.15, thickness=0.02):
        """Create small red cross marker"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = thickness
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Solid red
        
        x = point[1] * map_info.resolution + map_info.origin.position.x
        y = point[0] * map_info.resolution + map_info.origin.position.y
        z = 0.15
        
        p1 = Point(x=x-size/2, y=y-size/2, z=z)
        p2 = Point(x=x+size/2, y=y+size/2, z=z)
        p3 = Point(x=x-size/2, y=y+size/2, z=z)
        p4 = Point(x=x+size/2, y=y-size/2, z=z)
        
        marker.points.extend([p1, p2, p3, p4])
        
        return marker