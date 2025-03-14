import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN

class FrontierDetector:
    def __init__(self, min_frontier_size, clustering_eps):
        self.min_frontier_size = min_frontier_size
        self.clustering_eps = clustering_eps
        self.wall_thickness = 10  # Number of dilations for walls

    def detect_frontiers(self, map_data):
        free_space = (map_data == 0).astype(np.uint8)
        unknown_space = (map_data == -1).astype(np.uint8)
        occupied_space = (map_data > 50).astype(np.uint8)

        # Create kernel for dilation
        kernel = np.ones((3, 3), np.uint8)
        
        # Dilate free space once
        free_space_dilated = ndimage.binary_dilation(free_space, kernel)
        
        # Dilate occupied space multiple times for thicker walls
        occupied_space_dilated = occupied_space
        for _ in range(self.wall_thickness):
            occupied_space_dilated = ndimage.binary_dilation(occupied_space_dilated, kernel)

        # Find frontiers avoiding thick walls
        frontier_cells = np.logical_and(
            np.logical_and(unknown_space, free_space_dilated),
            np.logical_not(occupied_space_dilated)
        )

        frontier_points = np.argwhere(frontier_cells)
        return self._cluster_frontiers(frontier_points)

    def _cluster_frontiers(self, frontier_points):
        if len(frontier_points) == 0:
            return np.array([])

        clustering = DBSCAN(
            eps=self.clustering_eps,
            min_samples=self.min_frontier_size
        ).fit(frontier_points)

        return frontier_points[clustering.labels_ != -1]