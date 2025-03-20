import numpy as np
from sklearn.cluster import DBSCAN
from nav_msgs.msg import OccupancyGrid

class GoalSelector:
    def __init__(self, information_radius, min_distance, max_distance, logger=None):
        # Existing parameters
        self.max_distance = max_distance
        self.min_cluster_size = 15
        self.clustering_eps = 4.0
        self.exploration_radius = information_radius
        self.distance_weight = 0.8
        self.unexplored_weight = 0.15
        self.obstacle_weight = 0.05
        self.logger = logger
        self.previous_centroid = None
        self.same_centroid_threshold = 2.5
        self.min_frontier_size = 20
        
        # Initialize clearance map attributes
        self.latest_clearance_map = None
        self.map_info = None
        self.clearance_sub = None
        
        # Change blacklist to single point
        self.failed_point = None  # Store only one failed point
        self.point_failure_threshold = 2.0  # Distance threshold to consider a point as "same"
        self.logger = logger
        self.logger.info('[Goal Selector] Goal selector initialized')

    def initialize_clearance_subscriber(self, node):
        """Initialize subscriber for clearance map"""
        self.clearance_sub = node.create_subscription(
            OccupancyGrid,
            '/clearance_map',
            self._on_clearance_map,
            10
        )
        if self.logger:
            self.logger.info('[Goal Selector] Initialized clearance map subscriber')

    def _on_clearance_map(self, msg):
        """Process incoming clearance map"""
        try:
            # Reshape clearance map data
            height, width = msg.info.height, msg.info.width
            clearance_data = np.array(msg.data).reshape((height, width))
            
            # Normalize to [0,1] range and store
            self.latest_clearance_map = clearance_data / 100.0
            self.map_info = msg.info
            
            if self.logger:
                self.logger.debug(
                    f'[Goal Selector] Updated clearance map {width}x{height}, '
                    f'range: [{np.min(self.latest_clearance_map):.2f}, '
                    f'{np.max(self.latest_clearance_map):.2f}]'
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f'[Goal Selector] Error processing clearance map: {str(e)}')

    def count_unexplored_cells(self, centroid, map_data):
        """Count unexplored cells in radius around centroid"""
        # Ensure integer coordinates
        y, x = np.round(centroid).astype(np.int32)
        height, width = map_data.shape
        count = 0
        
        radius = int(self.exploration_radius)
        
        # Check cells in square around centroid
        for i in range(max(0, y - radius), min(height, y + radius + 1)):
            for j in range(max(0, x - radius), min(width, x + radius + 1)):
                # Check if point is within circular radius
                if ((i - y)**2 + (j - x)**2) <= radius**2:
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

    def find_best_clearance_point(self, cluster_points, clearance_map):
        """Find the frontier point with highest clearance in the cluster"""
        try:
            if len(cluster_points) == 0 or clearance_map is None:
                return None
                
            # Get clearance values for all points in cluster
            clearance_values = []
            for point in cluster_points:
                y, x = int(point[0]), int(point[1])
                if (0 <= y < clearance_map.shape[0] and 
                    0 <= x < clearance_map.shape[1]):
                    clearance_values.append(clearance_map[y, x])
                else:
                    clearance_values.append(0)
                    
            # Select point with highest clearance
            best_idx = np.argmax(clearance_values)
            return cluster_points[best_idx], clearance_values[best_idx]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f'[Goal Selector] Error finding best clearance point: {str(e)}')
            return None, 0.0

    def is_point_blacklisted(self, point, map_info):
        """Check if a point is too close to the failed point"""
        if self.failed_point is None:
            return False
            
        point_world = self._to_world(point, map_info)
        distance = np.linalg.norm(point_world - self.failed_point)
        
        if distance < self.point_failure_threshold:
            if self.logger:
                self.logger.debug(
                    f'[Goal Selector] Point ({point[0]}, {point[1]}) is near '
                    f'failed point at ({self.failed_point[0]:.2f}, {self.failed_point[1]:.2f})'
                )
            return True
        return False

    def add_failed_point(self, world_point):
        """Store the failed point (in world coordinates)"""
        self.failed_point = np.array(world_point)
        if self.logger:
            self.logger.info(
                f'[Goal Selector] Updated failed point to '
                f'({world_point[0]:.2f}, {world_point[1]:.2f})'
            )

    def select_goal(self, frontier_points, map_data, map_info, robot_position, previous_goals=None):
        """Select best frontier using clearance information"""
        self.logger.info(f'[Goal Selector] Selecting goal from {len(frontier_points)} frontiers')
        try:
            if len(frontier_points) == 0:
                self.logger.warn('[Goal Selector] No frontier points available')
                return None, 0, [], None

            # Handle edge case with few frontier points
            if len(frontier_points) < self.min_cluster_size:
                self.logger.info(
                    f'[Goal Selector] Few frontier points ({len(frontier_points)}) - '
                    'using direct point selection'
                )
                return self._handle_few_frontiers(
                    frontier_points, 
                    map_data, 
                    map_info, 
                    robot_position
                )

            # Try clustering
            clustering = DBSCAN(
                eps=self.clustering_eps,
                min_samples=self.min_cluster_size
            ).fit(frontier_points)
            
            # Check if clustering succeeded
            valid_clusters = np.unique(clustering.labels_[clustering.labels_ != -1])
            if len(valid_clusters) == 0:
                self.logger.info('[Goal Selector] Clustering failed - trying direct selection')
                return self._handle_few_frontiers(
                    frontier_points, 
                    map_data, 
                    map_info, 
                    robot_position
                )

            clusters = self._cluster_frontiers(frontier_points)
            if not clusters:
                return None, 0, [], None

            # Store cluster labels for visualization
            cluster_labels = clustering.labels_

            best_point = None
            best_score = float('-inf')
            all_centroids = []

            self.logger.info(f"\n[Goal Selector] Evaluating {len(clusters)} clusters:")
            self.logger.info("[Goal Selector] Cluster | Size | Centroid (y,x) | Candidate (y,x) | Unexpl | Clear | Dist | Score")
            self.logger.info("[Goal Selector] " + "-" * 80)
            
            for cluster_idx, cluster in enumerate(clusters):
                # Calculate cluster centroid
                centroid = np.mean(cluster, axis=0)
                all_centroids.append(centroid)

                if self.latest_clearance_map is not None:
                    best_point_in_cluster, clearance_value = self.find_best_clearance_point(
                        cluster, self.latest_clearance_map
                    )
                    if best_point_in_cluster is not None:
                        candidate_point = best_point_in_cluster
                    else:
                        candidate_point = np.round(centroid).astype(np.int32)
                        clearance_value = 0
                else:
                    candidate_point = np.round(centroid).astype(np.int32)
                    clearance_value = 0

                # Skip blacklisted points
                if self.is_point_blacklisted(candidate_point, map_info):
                    status = "BLACKLISTED"
                    if self.logger:
                        self.logger.debug(f'[Goal Selector] Skipping blacklisted point: {candidate_point}')
                    continue

                # Calculate score components
                world_pos = self._to_world(candidate_point, map_info)
                distance = np.linalg.norm(robot_position - world_pos)
                unexplored = self.count_unexplored_cells(candidate_point, map_data)
                
                # Combined score
                score = (self.unexplored_weight * unexplored / 100.0 + 
                        self.obstacle_weight * clearance_value - 
                        self.distance_weight * distance / self.max_distance)

                # Format and log cluster info in one line
                cluster_info = (
                    f"{cluster_idx+1:^7} | "
                    f"{len(cluster):4} | "
                    f"({centroid[0]:5.1f},{centroid[1]:5.1f}) | "
                    f"({candidate_point[0]:3},{candidate_point[1]:3}) | "
                    f"{unexplored:6} | "
                    f"{clearance_value:5.2f} | "
                    f"{distance:4.1f} | "
                    f"{score:5.3f}"
                )
                self.logger.info("[Goal Selector] " + cluster_info + (" <= BEST" if score > best_score else ""))

                if score > best_score:
                    best_score = score
                    best_point = candidate_point.copy()

            if best_point is not None:
                self.logger.info("[Goal Selector] " + "-" * 80)
                self.logger.info(f"[Goal Selector] Selected: Cluster with point ({best_point[0]}, {best_point[1]}) - Score: {best_score:.3f}")
            else:
                self.logger.warn("[Goal Selector] No suitable candidate point found")
            
            # Return cluster labels as the fourth parameter
            return best_point, best_score, all_centroids, cluster_labels

        except Exception as e:
            self.logger.error(f'[Goal Selector] Goal selection failed: {str(e)}')
            return None, 0, [], None

    def _to_world(self, point, map_info):
        """Convert grid coordinates to world coordinates"""
        return np.array([
            point[1] * map_info.resolution + map_info.origin.position.x,
            point[0] * map_info.resolution + map_info.origin.position.y
        ])

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

    def _handle_few_frontiers(self, frontier_points, map_data, map_info, robot_position):
        """Handle case when there are too few frontier points for clustering"""
        best_point = None
        best_score = float('-inf')
        
        self.logger.info("[Goal Selector] Evaluating individual frontier points:")
        self.logger.info("[Goal Selector] Point (y,x) | Unexpl | Clear | Dist | Score")
        self.logger.info("[Goal Selector] " + "-" * 60)

        for point in frontier_points:
            # Skip blacklisted points
            if self.is_point_blacklisted(point, map_info):
                continue

            # Get clearance value
            if self.latest_clearance_map is not None:
                y, x = int(point[0]), int(point[1])
                if (0 <= y < self.latest_clearance_map.shape[0] and 
                    0 <= x < self.latest_clearance_map.shape[1]):
                    clearance_value = self.latest_clearance_map[y, x]
                else:
                    clearance_value = 0
            else:
                clearance_value = 0

            # Calculate score components
            world_pos = self._to_world(point, map_info)
            distance = np.linalg.norm(robot_position - world_pos)
            unexplored = self.count_unexplored_cells(point, map_data)
            
            # Combined score with adjusted weights for direct selection
            score = (
                1.2 * self.unexplored_weight * unexplored / 100.0 +  # Increased weight
                0.8 * self.obstacle_weight * clearance_value -       # Reduced weight
                1.0 * self.distance_weight * distance / self.max_distance
            )

            # Log point info
            point_info = (
                f"({point[0]:3},{point[1]:3}) | "
                f"{unexplored:6} | "
                f"{clearance_value:5.2f} | "
                f"{distance:4.1f} | "
                f"{score:5.3f}"
            )
            self.logger.info("[Goal Selector] " + point_info + (" <= BEST" if score > best_score else ""))

            if score > best_score:
                best_score = score
                best_point = point.copy()

        if best_point is not None:
            self.logger.info("[Goal Selector] " + "-" * 60)
            self.logger.info(
                f"[Goal Selector] Selected point ({best_point[0]}, {best_point[1]}) "
                f"with score: {best_score:.3f}"
            )
            # Create single cluster labels
            cluster_labels = np.zeros(len(frontier_points), dtype=np.int32)
            all_centroids = [np.mean(frontier_points, axis=0)]
            
            self.logger.info(
                "[Goal Selector] Treating all points as single cluster:"
                f"\n- Cluster size: {len(frontier_points)}"
                f"\n- Centroid: ({all_centroids[0][0]:.1f}, {all_centroids[0][1]:.1f})"
            )
            
            return best_point, best_score, all_centroids, cluster_labels
        else:
            self.logger.warn("[Goal Selector] No suitable point found in direct selection")
            return None, 0, [], None