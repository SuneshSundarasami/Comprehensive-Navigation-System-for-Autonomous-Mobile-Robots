import numpy as np
from sklearn.cluster import DBSCAN

class GoalSelector:
    def __init__(self, information_radius, min_distance, max_distance, logger=None):
        self.max_distance = 10.0  # Maximum distance to consider
        self.min_cluster_size = 5  # Increased minimum cluster size
        self.clustering_eps = 3.0  # Reduced epsilon for tighter clusters
        self.exploration_radius = 5  # Radius to check for unexplored cells
        self.distance_weight = 0.8  # Weight for distance score
        self.unexplored_weight = 0.2  # Weight for unexplored area score
        self.logger = logger 
        self.previous_centroid = None  # Add this line to track previous selection
        self.same_centroid_threshold = 2.0  # Distance threshold to consider centroids same

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

    def calculate_obstacle_score(self, centroid, map_data, max_check_radius=5):
        """Calculate score based on distance to nearest obstacle"""
        y, x = int(centroid[0]), int(centroid[1])
        height, width = map_data.shape
        min_dist = max_check_radius  # Initialize to maximum

        # Check cells in square around centroid
        for radius in range(1, max_check_radius + 1):
            found_obstacle = False
            # Check perimeter at this radius
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if abs(i) == radius or abs(j) == radius:  # Only check perimeter
                        check_y, check_x = y + i, x + j
                        if (0 <= check_y < height and 
                            0 <= check_x < width and 
                            map_data[check_y, check_x] > 50):
                            min_dist = min(min_dist, np.sqrt(i*i + j*j))
                            found_obstacle = True
                            break
                if found_obstacle:
                    break
            if found_obstacle:
                break
        
        # Convert to score (closer to obstacles = lower score)
        return min_dist / max_check_radius

    def select_goal(self, frontier_points, map_data, map_info, robot_position, previous_goals=None):
        """Select best frontier centroid based on distance and information gain."""
        try:
            if len(frontier_points) == 0:
                return None, 0, []

            # Cluster the frontier points with adjusted parameters
            clustering = DBSCAN(
                eps=3.0,               # Smaller eps for more clusters
                min_samples=5,         # Reasonable minimum points
                metric='euclidean',
                algorithm='ball_tree',
                n_jobs=-1
            ).fit(frontier_points)
            
            # Debug clustering results
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            self.logger.info(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
            
            # Get cluster labels and centroids
            unique_labels = np.unique(clustering.labels_[clustering.labels_ != -1])
            centroids = []
            scores = []
            
            self.logger.info(f"Evaluating {len(unique_labels)} clusters:")
            
            # Store all valid centroids and scores
            valid_centroids = []
            valid_scores = []
            all_centroids = []  # Store all centroids for visualization
            
            for label in unique_labels:
                cluster_points = frontier_points[clustering.labels_ == label]
                if len(cluster_points) < self.min_cluster_size:
                    continue
                    
                centroid = np.mean(cluster_points, axis=0)
                all_centroids.append(centroid)  # Add to all centroids list
                
                # Calculate obstacle proximity score
                obstacle_score = self.calculate_obstacle_score(centroid, map_data)
                if obstacle_score < 0.3:  # Skip if too close to obstacles
                    self.logger.info(f"Skipping cluster - too close to obstacles (score: {obstacle_score:.2f})")
                    continue
                
                # Convert centroid to world coordinates
                world_x = centroid[1] * map_info.resolution + map_info.origin.position.x
                world_y = centroid[0] * map_info.resolution + map_info.origin.position.y
                
                # Calculate metrics
                distance = np.linalg.norm(robot_position - np.array([world_x, world_y]))
                unexplored_count = self.count_unexplored_cells(centroid, map_data)
                
                # Skip if too far
                if distance > self.max_distance:
                    continue
                    
                # Calculate combined score including obstacle proximity
                distance_score = 1.0 - (distance / self.max_distance)
                unexplored_score = min(1.0, unexplored_count / (np.pi * self.exploration_radius**2))
                score = (0.3 * distance_score + 
                        0.4 * unexplored_score + 
                        0.3 * obstacle_score)  # Added obstacle score component
                
                # Check if this centroid is too close to the previous one
                if self.previous_centroid is not None:
                    prev_dist = np.linalg.norm(centroid - self.previous_centroid)
                    if prev_dist < self.same_centroid_threshold:
                        self.logger.info(
                            f"Skipping centroid at ({world_x:.2f}, {world_y:.2f}) - too close to previous"
                        )
                        continue
                
                self.logger.info(
                    f"Cluster at ({world_x:.2f}, {world_y:.2f}): "
                    f"distance={distance:.1f}, unexplored={unexplored_count}, "
                    f"obstacle_score={obstacle_score:.2f}, total_score={score:.3f}"
                )
                
                valid_centroids.append(centroid)
                valid_scores.append(score)
            
            if not valid_centroids:
                # If no other options, clear previous and allow reuse
                self.previous_centroid = None
                return None, 0, all_centroids  # Return all_centroids even if no valid ones
                
            # Select best centroid
            best_idx = np.argmax(valid_scores)
            selected_centroid = valid_centroids[best_idx]
            
            # Store selected centroid for next iteration
            self.previous_centroid = selected_centroid.copy()
            
            return selected_centroid, -valid_scores[best_idx], all_centroids  # Return all centroids

        except Exception as e:
            self.logger.error(f'Goal selection failed: {str(e)}')
            return None, 0, []

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