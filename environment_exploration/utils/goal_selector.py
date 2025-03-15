import numpy as np
from sklearn.cluster import DBSCAN

class GoalSelector:
    def __init__(self, information_radius, min_distance, max_distance, logger=None):
        self.max_distance = 5.0  # Maximum distance to consider
        self.min_cluster_size = 5
        self.clustering_eps = 2.0
        self.exploration_radius = 10  # Radius to check for unexplored cells
        self.distance_weight = 0.4  # Weight for distance score
        self.unexplored_weight = 0.6  # Weight for unexplored area score
        self.logger = logger 

    def count_unexplored_cells(self, centroid, map_data):
        """Count unexplored cells in radius around centroid"""
        y, x = int(centroid[0]), int(centroid[1])
        height, width = map_data.shape
        count = 0
        
        # Check cells in square around centroid
        for i in range(max(0, y - self.exploration_radius), 
                      min(height, y + self.exploration_radius + 1)):
            for j in range(max(0, x - self.exploration_radius), 
                         min(width, x + self.exploration_radius + 1)):
                # Check if point is within circular radius
                if ((i - y)**2 + (j - x)**2) <= self.exploration_radius**2:
                    if map_data[i, j] == -1:  # Unexplored cell
                        count += 1
        
        return count

    def select_goal(self, frontier_points, map_data, map_info, robot_position, previous_goals):
        """Select frontier point based on distance and unexplored area."""
        if len(frontier_points) == 0:
            return None, float('-inf')

        # Cluster frontiers
        clusters = self._cluster_frontiers(frontier_points)
        if not clusters:
            return None, float('-inf')

        # Evaluate each cluster
        best_score = float('-inf')
        best_centroid = None
        max_distance = 0
        max_unexplored = 0

        # First pass to get normalization factors
        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            distance = np.linalg.norm(centroid - robot_position)
            unexplored = self.count_unexplored_cells(centroid, map_data)
            
            max_distance = max(max_distance, distance)
            max_unexplored = max(max_unexplored, unexplored)

        # Second pass to compute normalized scores
        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            distance = np.linalg.norm(centroid - robot_position)
            unexplored = self.count_unexplored_cells(centroid, map_data)

            # Convert centroid to map coordinates
            map_x = centroid[1] * map_info.resolution + map_info.origin.position.x
            map_y = centroid[0] * map_info.resolution + map_info.origin.position.y
        
            
            # Normalize scores between 0 and 1
            distance_score = 1.0 - (distance / max_distance if max_distance > 0 else 0)
            unexplored_score = unexplored / max_unexplored if max_unexplored > 0 else 0
            
            # Combine scores
            total_score = (self.distance_weight * distance_score + 
                         self.unexplored_weight * unexplored_score)
            
            self.logger.info(
                f"Cluster at ({map_x:.2f}, {map_y:.2f}): "
                f"distance={distance:.1f}, unexplored={unexplored}, "
                f"score={total_score:.3f}"
            )
            
            if total_score > best_score:
                best_score = total_score
                best_centroid = centroid

        if best_centroid is None:
            return None, float('-inf')

        return best_centroid, best_score

    def _cluster_frontiers(self, frontier_points):
        """Simple clustering of frontier points."""
        if len(frontier_points) < self.min_cluster_size:
            return []

        clustering = DBSCAN(
            eps=self.clustering_eps,
            min_samples=self.min_cluster_size
        ).fit(frontier_points)

        # Group points by cluster
        clusters = []
        unique_labels = np.unique(clustering.labels_)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise

        for label in unique_labels:
            cluster_points = frontier_points[clustering.labels_ == label]
            clusters.append(cluster_points)

        return clusters