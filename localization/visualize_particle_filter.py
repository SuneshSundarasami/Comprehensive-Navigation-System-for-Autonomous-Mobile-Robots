import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Point
from std_msgs.msg import ColorRGBA
import numpy as np
import math

class ParticleFilterVisualizer(Node):
    def __init__(self):
        super().__init__('particle_filter_visualizer')
        
        # Create subscribers
        self.particles_sub = self.create_subscription(
            PoseArray,
            'particles',
            self.particles_callback,
            10
        )
        
        # Create publishers for visualization
        self.marker_pub = self.create_publisher(
            MarkerArray,
            'particle_markers',
            10
        )
        
        # Visualization parameters
        self.particle_color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)  # Blue particles
        self.best_particle_color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red best estimate
        self.marker_lifetime = 0.1  # Markers last for 0.1 seconds
        
        self.get_logger().info('Particle Filter Visualizer initialized')

    def particles_callback(self, msg):
        """Create and publish visualization markers for particles"""
        marker_array = MarkerArray()
        
        # Create marker for particle arrows
        particles_marker = Marker()
        particles_marker.header = msg.header
        particles_marker.ns = 'particles'
        particles_marker.id = 0
        particles_marker.type = Marker.ARROW
        particles_marker.action = Marker.ADD
        particles_marker.scale.x = 0.3  # Arrow length
        particles_marker.scale.y = 0.1  # Arrow width
        particles_marker.scale.z = 0.1  # Arrow height
        particles_marker.color = self.particle_color
        particles_marker.lifetime.sec = 0
        particles_marker.lifetime.nanosec = int(self.marker_lifetime * 1e9)
        
        # Add arrows for each particle
        for pose in msg.poses:
            # Start point
            start_point = Point(
                x=pose.position.x,
                y=pose.position.y,
                z=0.1
            )
            particles_marker.points.append(start_point)
            
            # Calculate yaw from quaternion
            yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)
            
            # End point for arrow
            end_point = Point(
                x=pose.position.x + 0.2 * math.cos(yaw),
                y=pose.position.y + 0.2 * math.sin(yaw),
                z=0.1
            )
            particles_marker.points.append(end_point)
        
        marker_array.markers.append(particles_marker)
        
        # Create marker for variance visualization
        if len(msg.poses) > 0:
            positions = np.array([[p.position.x, p.position.y] for p in msg.poses])
            variance = np.var(positions, axis=0)
            variance_marker = self.create_variance_marker(
                positions.mean(axis=0),
                variance,
                msg.header
            )
            marker_array.markers.append(variance_marker)
        
        # Publish all markers
        self.marker_pub.publish(marker_array)

    def create_variance_marker(self, mean_pos, variance, header):
        """Create a marker showing particle distribution variance"""
        marker = Marker()
        marker.header = header
        marker.ns = 'variance'
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Position at mean of particles
        marker.pose.position.x = mean_pos[0]
        marker.pose.position.y = mean_pos[1]
        marker.pose.position.z = 0.1
        
        # Scale based on variance
        variance_scale = np.sqrt(variance)
        marker.scale.x = max(0.1, variance_scale[0])
        marker.scale.y = max(0.1, variance_scale[1])
        marker.scale.z = 0.1
        
        # Color based on variance (more red = more uncertain)
        certainty = np.exp(-np.mean(variance))
        marker.color = ColorRGBA(
            r=1.0 - certainty,
            g=certainty,
            b=0.0,
            a=0.3
        )
        
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = int(self.marker_lifetime * 1e9)
        
        return marker

def main(args=None):
    rclpy.init(args=args)
    visualizer = ParticleFilterVisualizer()
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()