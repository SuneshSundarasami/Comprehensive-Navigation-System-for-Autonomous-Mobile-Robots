import numpy as np

class MeasurementModel:
    def __init__(self):
        self.max_range = 3.0  # meters
        self.hit_score = 1.0
        self.miss_penalty = 0.5
        self.neighbor_size = 2  # Check ±2 cells in each direction
        
    def check_neighborhood(self, map_data, x, y):
        """Check neighboring cells for occupancy"""
        height, width = map_data.shape
        
        # Define neighborhood bounds
        x_min = max(0, x - self.neighbor_size)
        x_max = min(width - 1, x + self.neighbor_size)
        y_min = max(0, y - self.neighbor_size)
        y_max = min(height - 1, y + self.neighbor_size)
        
        # Check neighborhood for occupied cells
        neighborhood = map_data[y_min:y_max+1, x_min:x_max+1]
        has_occupied = np.any(neighborhood == 100)
        has_free = np.any(neighborhood == 0)
        
        return has_occupied, has_free
    
    def compute_likelihood(self, particle, scan_points, map_data, map_info):
        """
        Lenient measurement model that checks neighborhood of scan points
        """
        if len(scan_points) < 3:
            return 0.0
            
        # Transform scan points to map frame
        cos_theta = np.cos(particle[2])
        sin_theta = np.sin(particle[2])
        R = np.array([[cos_theta, -sin_theta],
                     [sin_theta, cos_theta]])
        
        world_points = np.dot(scan_points, R.T) + particle[:2]
        
        # Convert to map coordinates
        map_points = np.zeros_like(world_points, dtype=int)
        map_points[:, 0] = ((world_points[:, 0] - map_info.origin.position.x) 
                           / map_info.resolution).astype(int)
        map_points[:, 1] = ((world_points[:, 1] - map_info.origin.position.y) 
                           / map_info.resolution).astype(int)
        
        # Check which points are within map bounds
        valid = ((map_points[:, 0] >= 0) & (map_points[:, 0] < map_data.shape[1]) &
                (map_points[:, 1] >= 0) & (map_points[:, 1] < map_data.shape[0]))
        
        if not np.any(valid):
            return 0.0
        
        # Count hits and misses with neighborhood checking
        score = 0.0
        num_valid = 0
        
        for x, y, is_valid in zip(map_points[:, 0], map_points[:, 1], valid):
            if is_valid:
                num_valid += 1
                has_occupied, has_free = self.check_neighborhood(map_data, x, y)
                
                if has_occupied:  # Hit near occupied cell
                    score += self.hit_score
                elif has_free:    # Miss but free cells nearby
                    score -= self.miss_penalty
        
        if num_valid == 0:
            return 0.0
            
        # Normalize score by number of valid points
        likelihood = max(0.0, score / num_valid)
        return likelihood