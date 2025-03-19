import numpy as np

class ExplorationProgress:
    def __init__(self, logger=None):
        self.logger = logger
        self.selected_cluster_points = None
        self.monitoring_threshold = 0.75  # 75% mapped threshold
        
        if self.logger:
            self.logger.info('[Progress Monitor] Initialized with threshold: 75%')

    def update_cluster(self, points):
        """Update the points being monitored"""
        self.selected_cluster_points = points
        if self.logger:
            self.logger.info(
                f'[Progress Monitor] Updated cluster points:'
                f'\n- Total points: {len(points)}'
            )

    def check_progress(self, current_map):
        """Check if cluster is sufficiently explored"""
        if self.selected_cluster_points is None or len(self.selected_cluster_points) == 0:
            return False

        try:
            mapped_count = 0
            total_points = len(self.selected_cluster_points)

            for point in self.selected_cluster_points:
                y, x = int(point[0]), int(point[1])
                if 0 <= y < current_map.shape[0] and 0 <= x < current_map.shape[1]:
                    if current_map[y, x] != -1:  # Point is now mapped
                        mapped_count += 1

            progress = mapped_count / total_points
            
            if self.logger:
                self.logger.debug(
                    f'[Progress Monitor] Mapping progress:'
                    f'\n- Mapped points: {mapped_count}/{total_points}'
                    f'\n- Progress: {progress:.1%}'
                )

            return progress >= self.monitoring_threshold

        except Exception as e:
            if self.logger:
                self.logger.error(f'[Progress Monitor] Error checking progress: {str(e)}')
            return False