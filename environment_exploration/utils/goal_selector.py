import numpy as np
from sklearn.cluster import DBSCAN

class GoalSelector:
    def __init__(self, information_radius, min_distance, max_distance):
        self.max_distance = 5.0  # Maximum distance to consider
        self.min_cluster_size = 5
        self.clustering_eps = 2.0

    def select_goal(self, frontier_points, map_data, map_info, robot_position, previous_goals):
        """Select the closest valid frontier point as goal."""
        if len(frontier_points) == 0:
            return None, float('-inf')

        # Cluster frontiers
        clusters = self._cluster_frontiers(frontier_points)
        if not clusters:
            return None, float('-inf')

        # Find closest centroid
        min_distance = float('inf')
        best_centroid = None

        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            distance = np.linalg.norm(centroid - robot_position)
            
            print(f"Evaluating cluster - Distance: {distance:.2f}")
            
            # Update if this is the closest valid centroid
            if distance < min_distance:
                min_distance = distance
                best_centroid = centroid

        if best_centroid is None:
            return None, float('-inf')

        print(f"Selected goal at distance: {min_distance:.2f}")
        return best_centroid, -min_distance  # Negative distance as score

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