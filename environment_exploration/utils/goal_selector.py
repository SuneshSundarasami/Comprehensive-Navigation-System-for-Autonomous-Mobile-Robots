import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

class GoalSelector:
    def __init__(self, information_radius, min_distance, max_distance):
        self.information_radius = information_radius
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.min_cluster_size = 5  # Minimum points for a cluster
        self.clustering_eps = 2.0   # Maximum distance between points in a cluster

    def select_goal(self, frontier_points, map_data, map_info, robot_position, previous_goals):
        if len(frontier_points) == 0:
            return None, float('-inf')

        clusters = self._cluster_frontiers(frontier_points)
        if not clusters:
            return None, float('-inf')

        scores = []
        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            score = self._evaluate_cluster(centroid, cluster, map_data, map_info, robot_position, previous_goals)
            scores.append((centroid, score))

        if not scores:
            return None, float('-inf')

        best_centroid, best_score = max(scores, key=lambda x: x[1])
        return best_centroid, best_score

    def _cluster_frontiers(self, frontier_points):
        """Cluster frontier points using DBSCAN."""
        if len(frontier_points) < self.min_cluster_size:
            return []

        # Perform clustering
        clustering = DBSCAN(
            eps=self.clustering_eps,
            min_samples=self.min_cluster_size
        ).fit(frontier_points)

        # Get unique cluster labels (excluding noise points labeled as -1)
        unique_labels = np.unique(clustering.labels_)
        unique_labels = unique_labels[unique_labels != -1]

        # Group points by cluster
        clusters = []
        for label in unique_labels:
            cluster_points = frontier_points[clustering.labels_ == label]
            clusters.append(cluster_points)

        return clusters

    def _evaluate_cluster(self, centroid, cluster_points, map_data, map_info, robot_position, previous_goals):
        """Evaluate cluster based on multiple criteria."""
        # Add debug logging
        print(f"Evaluating cluster with {len(cluster_points)} points")
        
        # Distance cost
        distance = np.linalg.norm(centroid - robot_position)
        print(f"Distance to robot: {distance}")
        
        # Relax distance constraints
        if distance < self.min_distance * 0.5 or distance > self.max_distance * 2.0:
            print(f"Distance {distance} outside bounds [{self.min_distance}, {self.max_distance}]")
            return float('-inf')
        
        # Normalize distance cost between 0 and 1
        distance_cost = 1.0 - min(distance / self.max_distance, 1.0)
        print(f"Distance cost: {distance_cost}")

        # Information gain (amount of unknown space)
        info_gain = self._calculate_information_gain(centroid, map_data, map_info)
        print(f"Information gain: {info_gain}")

        # Novelty (distance from previous goals)
        novelty = 1.0
        if previous_goals and len(previous_goals) > 0:
            distances = cdist([centroid], previous_goals)
            min_dist_to_previous = np.min(distances)
            novelty = min(min_dist_to_previous / self.max_distance, 1.0)
        print(f"Novelty: {novelty}")

        # Cluster size weight (normalize by max expected size)
        max_expected_size = 50  # Adjust based on your map scale
        size_weight = min(len(cluster_points) / max_expected_size, 1.0)
        print(f"Size weight: {size_weight}")

        # Combined score with normalized weights
        score = (0.3 * distance_cost + 
                0.3 * info_gain + 
                0.2 * novelty +
                0.2 * size_weight)
        
        print(f"Final score: {score}\n")
        
        # Ensure we don't return -inf unless absolutely necessary
        return max(score, -1e6)

    def _calculate_information_gain(self, point, map_data, map_info):
        """Calculate the potential information gain around a point."""
        x = int((point[0] - map_info.origin.position.x) / map_info.resolution)
        y = int((point[1] - map_info.origin.position.y) / map_info.resolution)
        radius = int(self.information_radius / map_info.resolution)

        # Ensure we stay within map bounds
        x_min = max(0, x - radius)
        x_max = min(map_data.shape[1], x + radius)
        y_min = max(0, y - radius)
        y_max = min(map_data.shape[0], y + radius)

        # Count unknown cells in the region
        window = map_data[y_min:y_max, x_min:x_max]
        unknown_count = np.sum(window == -1)
        total_cells = (y_max - y_min) * (x_max - x_min)

        return unknown_count / total_cells if total_cells > 0 else 0.0