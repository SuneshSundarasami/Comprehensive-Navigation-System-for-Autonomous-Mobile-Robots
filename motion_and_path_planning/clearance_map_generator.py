import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2

class ClearanceMapGenerator(Node):
    def __init__(self):
        super().__init__('clearance_map_generator')
        
        # Set up QoS profile for map subscription
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        # Create subscribers and publishers
        self.map_sub = self.create_subscription(
            OccupancyGrid, 
            '/map', 
            self.map_callback, 
            map_qos
        )
        self.clearance_pub = self.create_publisher(
            OccupancyGrid, 
            '/clearance_map', 
            10
        )
        
        self.map_info = None
        self.latest_map = None
        
    def map_callback(self, msg):
        """Process incoming map and generate clearance map"""
        try:
            self.map_info = msg.info
            raw_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
            
            # Generate clearance map
            clearance_map = self.compute_clearance_map(raw_map)
            
            # Publish clearance map
            if clearance_map is not None:
                clearance_msg = OccupancyGrid()
                clearance_msg.header = msg.header
                clearance_msg.info = msg.info
                
                # Scale clearance values to fit in [-128, 127] range
                # First normalize to [0, 1]
                normalized_map = clearance_map / np.max(clearance_map)
                # Then scale to [0, 100] for occupancy grid convention
                scaled_map = (normalized_map * 100).astype(np.int8)
                # Convert to list and ensure values are within bounds
                clearance_msg.data = scaled_map.flatten().tolist()
                
                self.clearance_pub.publish(clearance_msg)
                
                # Log successful update
                self.get_logger().debug('Published updated clearance map')
                
                # Save visualizations
                self.save_clearance_visualizations(clearance_map, raw_map)
                
        except Exception as e:
            self.get_logger().error(f'Error in map_callback: {str(e)}')

    def compute_clearance_map(self, raw_map):
        """Compute smooth clearance values considering both obstacles and unexplored regions"""
        try:
            # Create obstacle and unexplored masks
            obstacle_mask = (raw_map == 100)
            unexplored_mask = (raw_map == -1)
            
            # Compute distance transform for obstacles (primary influence)
            obstacle_clearance = distance_transform_edt(~obstacle_mask)
            
            # Compute distance transform for unexplored regions (secondary influence)
            unexplored_clearance = distance_transform_edt(~unexplored_mask)
            
            # Apply Gaussian smoothing to both clearance maps
            obstacle_clearance = cv2.GaussianBlur(obstacle_clearance, (7, 7), 1.5)
            unexplored_clearance = cv2.GaussianBlur(unexplored_clearance, (7, 7), 1.5)
            
            # Normalize both clearance maps to [0, 1] range with smooth transitions
            if np.max(obstacle_clearance) > 0:
                obstacle_clearance = obstacle_clearance / np.max(obstacle_clearance)
            if np.max(unexplored_clearance) > 0:
                unexplored_clearance = unexplored_clearance / np.max(unexplored_clearance)
            
            # Combine clearance maps with weights and smooth transition
            combined_clearance = (0.8 * obstacle_clearance + 
                                0.2 * unexplored_clearance)
            
            # Apply final smoothing
            combined_clearance = cv2.GaussianBlur(combined_clearance, (5, 5), 0.8)
            
            # Normalize final clearance map
            if np.max(combined_clearance) > 0:
                combined_clearance = combined_clearance / np.max(combined_clearance)
            
            self.get_logger().debug(
                f'Clearance map statistics:'
                f'\n- Obstacle influence: {np.mean(obstacle_clearance):.3f}'
                f'\n- Unexplored influence: {np.mean(unexplored_clearance):.3f}'
                f'\n- Combined mean: {np.mean(combined_clearance):.3f}'
            )
            
            return combined_clearance
            
        except Exception as e:
            self.get_logger().error(f"Exception in compute_clearance_map: {str(e)}")
            return None

    def save_clearance_visualizations(self, clearance_map, raw_map):
        """Generate and save visualization of clearance map"""
        try:
            # Create higher resolution visualization
            scale_factor = 2
            height, width = clearance_map.shape
            scaled_size = (width * scale_factor, height * scale_factor)
            
            # Scale up clearance map for smoother visualization
            clearance_img = cv2.resize(
                clearance_map,
                scaled_size,
                interpolation=cv2.INTER_CUBIC
            )
            
            # Normalize and apply colormap
            clearance_img = (clearance_img * 255).astype(np.uint8)
            clearance_colormap = cv2.applyColorMap(clearance_img, cv2.COLORMAP_JET)
            
            # Create high-res raw map visualization
            raw_map_vis = np.where(raw_map == 100, 0, 255).astype(np.uint8)
            raw_map_vis = cv2.resize(
                raw_map_vis,
                scaled_size,
                interpolation=cv2.INTER_NEAREST
            )
            raw_map_color = cv2.cvtColor(raw_map_vis, cv2.COLOR_GRAY2BGR)
            
            # Create smooth overlay
            overlay = cv2.addWeighted(clearance_colormap, 0.7, raw_map_color, 0.3, 0)
            
            # Save high-resolution images
            base_path = '/home/sunesh/ros2_ws/src/amr_project_amr_t04/clearance_maps/'
            cv2.imwrite(f'{base_path}clearance_map_smooth.png', clearance_colormap)
            cv2.imwrite(f'{base_path}overlay_clearance_map_smooth.png', overlay)
            
        except Exception as e:
            self.get_logger().error(f"Error saving visualizations: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ClearanceMapGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()