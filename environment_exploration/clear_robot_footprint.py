import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped ,Twist
import tf2_ros
import numpy as np
from math import pi

class ClearRobotFootprintNode(Node):
    def __init__(self):
        super().__init__('clear_robot_footprint')

      # Rotation control parameters
        self.start_time = None
        self.angular_speed = 0.75  # rad/s
        self.rotation_duration = 6 * pi / self.angular_speed  # Time for one rotation
        self.rotate_complete = False
        
        
        # Add cmd_vel publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        
        # Create rotation timer
        self.rotate_timer = self.create_timer(0.1, self.rotate_callback)
        
        
        # Subscribe to the map topic published by SLAM Toolbox
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            1)  # Use queue size 1 for latest message
        
        # Publisher for the modified map - publish to the same topic
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            'map',
            10)
        
        # TF listener to get robot position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Robot parameters
        self.robot_radius = 0.3  # Adjust based on your robot's size
        self.get_logger().info('Clear Robot Footprint node initialized - publishing to original map topic')
    
    def rotate_callback(self):
        if self.rotate_complete:
            return
            
        if self.start_time is None:
            self.start_time = self.get_clock().now()
            self.get_logger().info('Starting rotation...')
        
        # Calculate elapsed time
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        
        if elapsed >= self.rotation_duration:
            # Stop rotation
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)
            self.rotate_complete = True
            self.get_logger().info('Rotation complete!')
            return
        
        # Continue rotation
        twist_msg = Twist()
        twist_msg.angular.z = self.angular_speed
        self.cmd_vel_pub.publish(twist_msg)


    def map_callback(self, msg):
        try:
            # Convert (0,0) to grid coordinates
            origin_x = msg.info.origin.position.x
            origin_y = msg.info.origin.position.y
            
            # Convert world (0,0) to grid coordinates
            grid_x = int(abs(origin_x) / msg.info.resolution)
            grid_y = int(abs(origin_y) / msg.info.resolution)
            
            # Calculate radius in grid cells (make it a bit larger)
            radius_cells = int((self.robot_radius * 2) / msg.info.resolution)
            
            # Create a copy of the map data
            map_data = list(msg.data)
            
            # Clear cells around (0,0)
            width = msg.info.width
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    if dx*dx + dy*dy <= radius_cells*radius_cells:
                        idx = (grid_y + dy) * width + (grid_x + dx)
                        if 0 <= idx < len(map_data):
                            map_data[idx] = 0  # Set to free space
            
            # Create and publish cleared map
            cleared_map = OccupancyGrid()
            cleared_map.header = msg.header
            cleared_map.info = msg.info
            cleared_map.data = map_data
            self.map_pub.publish(cleared_map)
            
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ClearRobotFootprintNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()