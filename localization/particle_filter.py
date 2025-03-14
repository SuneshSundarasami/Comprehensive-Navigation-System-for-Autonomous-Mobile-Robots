import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist, PoseArray, Pose, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster
import math
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from .motion_model import MotionModel

class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')
        
        # Parameters
        self.declare_parameter('num_particles', 1000)  # Increased for better coverage
        self.declare_parameter('motion_update_rate', 10.0)  # Hz
        self.declare_parameter('measurement_update_rate', 5.0)  # Hz
        
        self.num_particles = self.get_parameter('num_particles').value
        self.motion_update_rate = self.get_parameter('motion_update_rate').value
        self.measurement_update_rate = self.get_parameter('measurement_update_rate').value
        
        self.motion_noise = [0.1, 0.1, 0.1]
        self.measurement_noise = 0.1
        self.resample_threshold = 0.5  # Threshold for effective particle ratio
        
        # Initialize parameters for measurement model
        self.max_scan_distance = 3.0  # Max distance to consider for scan matching
        self.scan_subsample = 5  # Use every nth scan ray
        
        self.local_init = True
        self.init_range = 0.5  # Small range to separate particles
        
        # Initialize motion model
        
        self.motion_model = MotionModel()
        
        # Initial poses for two particles
        self.init_pose = [-0.913, -4.88, 0.0]  # First particle at map origin
        
        # State variables
        self.particles = None  # [x, y, theta]
        self.weights = None
        self.map_data = None
        self.map_info = None
        self.last_cmd_time = None
        self.last_cmd_vel= None

        # Store latest messages
        self.latest_scan = None
        self.latest_cmd_vel = None
        self.last_motion_update = self.get_clock().now()

        map_qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
    )
    
        
        # Publishers
        self.particle_pub = self.create_publisher(PoseArray, 'particles', 10)
        self.estimate_pub = self.create_publisher(Marker, 'particle_estimate', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribers
        self.cmd_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, map_qos)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        
        # Timer for visualization
        self.create_timer(0.1, self.publish_visualization)

        # Create timers for regular updates
        self.motion_timer = self.create_timer(
            1.0/self.motion_update_rate, 
            self.motion_update
        )
        self.measurement_timer = self.create_timer(
            1.0/self.measurement_update_rate, 
            self.measurement_update
        )
        self.visualization_timer = self.create_timer(0.1, self.publish_visualization)

    def initialize_particles(self):
        """Initialize particles either locally or globally"""
        if self.map_data is None:
            self.get_logger().error('Map data not available for particle initialization')
            return
        
        if self.local_init:
            self.initialize_particles_local()
        else:
            self.initialize_particles_global()

    def initialize_particles_local(self):
        """Initialize particles near the initial pose"""
        self.particles = np.zeros((self.num_particles, 3))
        
        # Initialize particles in a small area
        x_range = [-1.0, 1.0]  # meters
        y_range = [-1.0, 1.0]  # meters
        theta_range = [-np.pi/4, np.pi/4]  # radians
        
        self.particles[:, 0] = np.random.uniform(x_range[0], x_range[1], self.num_particles)
        self.particles[:, 1] = np.random.uniform(y_range[0], y_range[1], self.num_particles)
        self.particles[:, 2] = np.random.uniform(theta_range[0], theta_range[1], self.num_particles)
        
        self.weights = np.ones(self.num_particles) / self.num_particles

    def initialize_particles_global(self):
        """Initialize particles across entire map"""
        # Original global initialization logic
        free_space = np.where(self.map_data == 0)
        if len(free_space[0]) == 0:
            self.get_logger().error('No free space found in map')
            return
        
        self.get_logger().info(f'Found {len(free_space[0])} free cells')
        
        # Initialize particles array
        self.particles = np.zeros((self.num_particles, 3))
        
        # Randomly select positions from free space
        indices = np.random.choice(len(free_space[0]), self.num_particles)
        
        # Convert map coordinates to world coordinates with proper origin offset
        # Note: In map coordinates, (0,0) is top-left corner
        # y-coordinate needs to be flipped because map origin is at bottom-left
        map_height = self.map_data.shape[0]
        
        # Convert coordinates with proper origin transformation
        self.particles[:, 0] = (free_space[1][indices] * self.map_info.resolution + 
                               self.map_info.origin.position.x)
        self.particles[:, 1] = ((map_height - free_space[0][indices]) * 
                               self.map_info.resolution + 
                               self.map_info.origin.position.y)
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)
        
        # Add boundary check
        map_bounds_x = [
            self.map_info.origin.position.x,
            self.map_info.origin.position.x + self.map_info.width * self.map_info.resolution
        ]
        map_bounds_y = [
            self.map_info.origin.position.y,
            self.map_info.origin.position.y + self.map_info.height * self.map_info.resolution
        ]
        
        # Filter particles to ensure they're within map bounds
        valid_particles = (
            (self.particles[:, 0] >= map_bounds_x[0]) & 
            (self.particles[:, 0] <= map_bounds_x[1]) & 
            (self.particles[:, 1] >= map_bounds_y[0]) & 
            (self.particles[:, 1] <= map_bounds_y[1])
        )
        
        self.particles = self.particles[valid_particles]
        self.num_particles = len(self.particles)
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        self.get_logger().info(f'Successfully initialized {self.num_particles} particles within map bounds')
        # Log map bounds and first few particles for debugging
        self.get_logger().info(f'Map bounds: X[{map_bounds_x[0]:.2f}, {map_bounds_x[1]:.2f}], ' +
                              f'Y[{map_bounds_y[0]:.2f}, {map_bounds_y[1]:.2f}]')
        for i in range(min(5, self.num_particles)):
            self.get_logger().info(f'Particle {i}: ({self.particles[i, 0]::.2f}, ' +
                                  f'{self.particles[i, 1]:.2f}, {self.particles[i, 2]:.2f})')

    def cmd_vel_callback(self, msg):
        """Store latest velocity command"""
        self.latest_cmd_vel = msg

    def scan_callback(self, msg):
        """Store latest scan"""
        self.latest_scan = msg

    def motion_update(self):
        """Update particles based on motion model at fixed frequency"""
        if self.particles is None or self.latest_cmd_vel is None:
            return
            
        current_time = self.get_clock().now()
        dt = (current_time - self.last_motion_update).nanoseconds / 1e9
        self.last_motion_update = current_time
        
        if dt < 0.01:
            return
        
        # Scale velocities to match robot motion
        velocity_scale = 0.1  # Reduce velocity magnitude
        linear_velocity = self.latest_cmd_vel.linear.x * velocity_scale
        angular_velocity = self.latest_cmd_vel.angular.z * velocity_scale
        
        # Debug velocity scaling
        self.get_logger().debug(
            f'Scaled velocities:'
            f'\n - Original linear: {self.latest_cmd_vel.linear.x:.3f}, scaled: {linear_velocity:.3f}'
            f'\n - Original angular: {self.latest_cmd_vel.angular.z:.3f}, scaled: {angular_velocity:.3f}'
            f'\n - dt: {dt:.3f}'
        )
        
        # Update particles using motion model with scaled velocities
        self.particles = self.motion_model.update_particles(
            self.particles,
            linear_velocity,
            angular_velocity,
            dt,
            self.motion_noise
        )

    def measurement_update(self):
        """Update particle weights based on measurement model at fixed frequency"""
        if self.particles is None or self.map_data is None or self.latest_scan is None:
            return
            
        msg = self.latest_scan
        epsilon = 1e-10
        
        # Pre-compute scan angles for efficiency
        angles = msg.angle_min + np.arange(0, len(msg.ranges), self.scan_subsample) * msg.angle_increment
        
        # Initialize weights array
        self.weights = np.zeros(len(self.particles)) + epsilon
        
        # Update weights for all particles
        for i, particle in enumerate(self.particles):
            num_matched = 0
            num_valid_beams = 0
            
            # Process scan beams
            for j, angle in enumerate(angles):
                range_reading = msg.ranges[j * self.scan_subsample]
                
                if range_reading > msg.range_min and range_reading < min(msg.range_max, self.max_scan_distance):
                    num_valid_beams += 1
                    
                    # Calculate expected scan endpoint in map coordinates
                    global_angle = particle[2] + angle
                    end_x = particle[0] + range_reading * np.cos(global_angle)
                    end_y = particle[1] + range_reading * np.sin(global_angle)
                    
                    # Convert to map coordinates
                    mx = int((end_x - self.map_info.origin.position.x) / self.map_info.resolution)
                    my = int((end_y - self.map_info.origin.position.y) / self.map_info.resolution)
                    
                    # Check if point is within map bounds and matches an obstacle
                    if (0 <= mx < self.map_data.shape[1] and 
                        0 <= my < self.map_data.shape[0]):
                        if self.map_data[my, mx] == 100:  # Hit matches obstacle
                            num_matched += 1
            
            # Calculate likelihood for this particle
            if num_valid_beams > 0:
                self.weights[i] = num_matched / num_valid_beams
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
            
            # Calculate effective particle ratio
            effective_particles = 1.0 / np.sum(np.square(self.weights))
            effective_ratio = effective_particles / len(self.particles)
            
            # Resample if effective ratio is below threshold
            if effective_ratio < self.resample_threshold:
                self.resample()

    def resample(self):
        """Systematic resampling"""
        positions = (np.random.random() + np.arange(self.num_particles)) / self.num_particles
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0
        
        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def publish_visualization(self):
        """Publish particles and best estimate"""
        if self.particles is None:
            self.get_logger().warn('Waiting for map data to initialize particles...')
            return
        
        # Publish particle cloud
        particle_msg = PoseArray()
        particle_msg.header.frame_id = 'map'
        particle_msg.header.stamp = self.get_clock().now().to_msg()
        
        for particle in self.particles:
            pose = Pose()
            pose.position.x = float(particle[0])
            pose.position.y = float(particle[1])
            pose.orientation.z = float(np.sin(particle[2] / 2))
            pose.orientation.w = float(np.cos(particle[2] / 2))
            particle_msg.poses.append(pose)
        
        self.particle_pub.publish(particle_msg)
        
        # Publish best estimate
        if len(self.particles) > 0:
            best_idx = np.argmax(self.weights)
            best_particle = self.particles[best_idx]
            
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.8  # Arrow length
            marker.scale.y = 0.05  # Arrow width
            marker.scale.z = 0.05  # Arrow height
            marker.color.a = 1.0
            marker.color.g = 1.0  # Green
            marker.pose.position.x = float(best_particle[0])
            marker.pose.position.y = float(best_particle[1])
            marker.pose.orientation.z = float(np.sin(best_particle[2] / 2))
            marker.pose.orientation.w = float(np.cos(best_particle[2] / 2))
            
            self.estimate_pub.publish(marker)
            # self.publish_transform(best_particle)

        self.get_logger().debug(f'Published {len(particle_msg.poses)} particles')

    def publish_transform(self, pose):
        """Publish transform from map to odom"""
        if not hasattr(self, 'current_odom'):
            return
            
        current_time = self.get_clock().now().to_msg()
        
        # Calculate transform from map to odom
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        
        # Calculate the transform that would move odom frame to align particle pose with map frame
        t.transform.translation.x = pose[0] - self.current_odom.pose.pose.position.x
        t.transform.translation.y = pose[0] - self.current_odom.pose.pose.position.y
        t.transform.translation.z = 0.0
        
        # Calculate rotation difference
        _, _, current_yaw = self.euler_from_quaternion(self.current_odom.pose.pose.orientation)
        rotation_diff = pose[2] - current_yaw
        
        # Convert to quaternion
        t.transform.rotation.z = np.sin(rotation_diff / 2.0)
        t.transform.rotation.w = np.cos(rotation_diff / 2.0)
        
        # Publish transform
        self.tf_broadcaster.sendTransform(t)

    def euler_from_quaternion(self, quaternion):
        """Convert quaternion to euler angles"""
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def map_callback(self, msg):
        """Store map data and initialize particles"""
        self.get_logger().info('Received map data')
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        
        # Debug map properties
        self.get_logger().info(f'Map dimensions: {msg.info.width}x{msg.info.height}')
        self.get_logger().info(f'Map origin: ({msg.info.origin.position.x}, {msg.info.origin.position.y})')
        self.get_logger().info(f'Map resolution: {msg.info.resolution}')
        
        # Initialize particles if not already initialized
        if self.particles is None:
            self.initialize_particles()

    def odom_callback(self, msg):
        """Store odometry data"""
        self.current_odom = msg

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()