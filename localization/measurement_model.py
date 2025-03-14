import numpy as np

class MeasurementModel:
    def __init__(self):
        self.max_range = 3.0  # meters
        self.hit_score = 1.0
        self.miss_penalty = 0.5
        
    def compute_likelihood(self, particle, scan_points, map_data, map_info):
        """
        Simple measurement model that checks if scan points hit occupied cells
        
        Args:
            particle: [x, y, theta] pose of particle
            scan_points: Nx2 array of scan points in robot frame
            map_data: 2D occupancy grid (0=free, 100=occupied, -1=unknown)
            map_info: Map metadata including resolution and origin
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
        
        # Count hits and misses
        score = 0.0
        num_valid = 0
        
        for x, y, is_valid in zip(map_points[:, 0], map_points[:, 1], valid):
            if is_valid:
                num_valid += 1
                if map_data[y, x] == 100:  # Hit on occupied cell
                    score += self.hit_score
                elif map_data[y, x] == 0:  # Miss (hit on free cell)
                    score -= self.miss_penalty
        
        if num_valid == 0:
            return 0.0
            
        # Normalize score by number of valid points
        likelihood = max(0.0, score / num_valid)
        return likelihood