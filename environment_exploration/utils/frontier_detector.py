import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
import rclpy

class FrontierDetector:
    def __init__(self):
        self.logger = rclpy.logging.get_logger('frontier_detector')

    def detect_frontiers(self, map_data):
        """Detect all frontier cells including map edges"""
        try:
            # Basic space classification
            free_space = (map_data == 0)
            unknown_space = (map_data == -1)
            
            # Get dimensions
            height, width = map_data.shape
            frontier_cells = np.zeros_like(free_space, dtype=bool)
            
            # Check all cells including edges
            for y in range(0, height):
                for x in range(0, width):
                    # Skip obstacles
                    if map_data[y, x] > 50:
                        continue
                        
                    # Define neighbors based on position
                    neighbors = []
                    
                    # Add valid neighbors checking bounds
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1),   # Direct neighbors
                                 (-1,-1), (-1,1), (1,-1), (1,1)]:  # Diagonal neighbors
                        ny, nx = y + dy, x + dx
                        
                        # Consider out-of-bounds as unknown
                        if 0 <= ny < height and 0 <= nx < width:
                            neighbors.append((ny, nx))
                        else:
                            # Edge cell with free neighbor is frontier
                            if free_space[y, x]:
                                frontier_cells[y, x] = True
                                break
                    
                    # If not already marked as frontier, check neighbors
                    if not frontier_cells[y, x]:
                        if free_space[y, x]:
                            # Free cell with unknown neighbor is frontier
                            if any(not (0 <= ny < height and 0 <= nx < width) or 
                                   unknown_space[ny, nx] for ny, nx in neighbors):
                                frontier_cells[y, x] = True
                                
                        elif unknown_space[y, x]:
                            # Unknown cell with free neighbor is frontier
                            if any(free_space[ny, nx] for ny, nx in neighbors):
                                frontier_cells[y, x] = True
            
            # Convert to points
            frontier_points = np.transpose(np.nonzero(frontier_cells))
            
            self.logger.info(
                f'Frontier detection results:\n'
                f'- Total points: {len(frontier_points)}\n'
                f'- Free cells: {np.sum(free_space)}\n'
                f'- Unknown cells: {np.sum(unknown_space)}'
            )
            
            return frontier_points

        except Exception as e:
            self.logger.error(f'Error in frontier detection: {str(e)}')
            return np.array([])