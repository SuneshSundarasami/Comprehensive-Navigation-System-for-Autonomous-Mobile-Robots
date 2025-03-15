import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN

class FrontierDetector:
    def __init__(self, min_frontier_size=3, clustering_eps=3.0):
        self.min_frontier_size = min_frontier_size
        self.clustering_eps = clustering_eps
        self.wall_thickness = 2  # Reduced wall thickness
        self.kernel = np.ones((3, 3), np.uint8)  # Cache kernel

    def detect_frontiers(self, map_data):
        """Detect frontiers with optimized processing"""
        # Use boolean arrays instead of uint8 for faster processing
        free_space = (map_data == 0)
        unknown_space = (map_data == -1)
        occupied_space = (map_data > 50)

        # Quick check if there are any unknown cells
        if not np.any(unknown_space):
            return np.array([])
        
        # Optimize dilations using pre-cached kernel
        free_space_dilated = ndimage.binary_dilation(
            free_space, 
            self.kernel,
            iterations=1
        )
        
        # Single-pass dilation for occupied space
        occupied_space_dilated = ndimage.binary_dilation(
            occupied_space, 
            self.kernel,
            iterations=self.wall_thickness
        )

        # Find frontiers (vectorized operations)
        frontier_cells = unknown_space & free_space_dilated & ~occupied_space_dilated

        # Quick exit if no frontier cells
        if not np.any(frontier_cells):
            return np.array([])

        # Efficient noise removal
        frontier_cells = ndimage.binary_opening(
            frontier_cells, 
            structure=self.kernel
        )

        # Convert to points more efficiently
        frontier_points = np.transpose(np.nonzero(frontier_cells))
        
        # Early exit if too few points
        if len(frontier_points) < self.min_frontier_size:
            return np.array([])

        return self._cluster_frontiers(frontier_points)

    def _cluster_frontiers(self, frontier_points):
        """Cluster frontiers with optimized parameters"""
        # Early exit check
        if len(frontier_points) == 0:
            return np.array([])

        # Use DBSCAN with optimized parameters
        clustering = DBSCAN(
            eps=self.clustering_eps,
            min_samples=self.min_frontier_size,
            metric='euclidean',
            algorithm='ball_tree',  # Better for 2D data
            n_jobs=-1  # Use all CPU cores
        ).fit(frontier_points)

        # Get valid clusters more efficiently
        valid_mask = clustering.labels_ != -1
        if not np.any(valid_mask):
            return np.array([])

        return frontier_points[valid_mask]