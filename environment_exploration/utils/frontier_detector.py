import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
import rclpy

class FrontierDetector:
    def __init__(self, min_frontier_size=3, clustering_eps=3.0):
        self.wall_thickness = 2
        self.kernel = np.ones((3, 3), np.uint8)
        self.logger = rclpy.logging.get_logger('frontier_detector')

    def detect_frontiers(self, map_data):
        """Detect frontiers with optimized processing"""
        self.logger.info(f'Processing map of shape {map_data.shape}')
        
        # Use boolean arrays for faster processing
        free_space = (map_data == 0)
        unknown_space = (map_data == -1)
        occupied_space = (map_data > 50)

        self.logger.info(f'Found {np.sum(free_space)} free cells, {np.sum(unknown_space)} unknown cells')

        if not np.any(unknown_space):
            self.logger.warn('No unknown space found, returning empty array')
            return np.array([])
        
        # Process spaces and track timing
        start_time = rclpy.clock.Clock().now()
        
        free_space_dilated = ndimage.binary_dilation(
            free_space, 
            self.kernel,
            iterations=1
        )
        
        occupied_space_dilated = ndimage.binary_dilation(
            occupied_space, 
            self.kernel,
            iterations=self.wall_thickness
        )

        # Find frontiers
        frontier_cells = unknown_space & free_space_dilated & ~occupied_space_dilated
        frontier_count = np.sum(frontier_cells)
        self.logger.info(f'Found {frontier_count} initial frontier cells')

        if not np.any(frontier_cells):
            self.logger.warn('No frontier cells found after initial detection')
            return np.array([])

        # Convert to points and return
        frontier_points = np.transpose(np.nonzero(frontier_cells))
        
        # Log processing time
        end_time = rclpy.clock.Clock().now()
        duration = (end_time - start_time).nanoseconds / 1e9
        self.logger.info(f'Frontier detection took {duration:.3f} seconds')

        return frontier_points