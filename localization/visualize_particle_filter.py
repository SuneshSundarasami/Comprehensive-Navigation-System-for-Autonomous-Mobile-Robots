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
        self.best_particle_color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green best estimate
        self.marker_lifetime = 0.1  # Markers last for 0.1 seconds
        
        # Best particle arrow parameters
        self.best_arrow_scale = Point(x=1.3, y=0.5, z=0.3)  # Bigger arrow for best estimate
        self.particle_arrow_scale = Point(x=0.3, y=0.1, z=0.1)  # Regular particles
        
        self.get_logger().info('Particle Filter Visualizer initialized')

    def particles_callback(self, msg):
        """Create and publish visualization markers for particles"""
        marker_array = MarkerArray()
        
        if len(msg.poses) == 0:
            return
            
        # Find the best particle (using mean position as estimate)
        positions = np.array([[p.position.x, p.position.y] for p in msg.poses])
        mean_pos = positions.mean(axis=0)
        best_idx = np.argmin(np.sum((positions - mean_pos)**2, axis=1))
        best_pose = msg.poses[best_idx]
        
        # Create marker for regular particle arrows
        particles_marker = Marker()
        particles_marker.header = msg.header
        particles_marker.ns = 'particles'
        particles_marker.id = 0
        particles_marker.type = Marker.ARROW
        particles_marker.action = Marker.ADD
        particles_marker.scale = self.particle_arrow_scale
        particles_marker.color = self.particle_color
        particles_marker.lifetime.sec = 0
        particles_marker.lifetime.nanosec = int(self.marker_lifetime * 1e9)
        
        # Add arrows for each particle except best
        for i, pose in enumerate(msg.poses):
            if i == best_idx:
                continue
            
            start_point = Point(x=pose.position.x, y=pose.position.y, z=0.1)
            particles_marker.points.append(start_point)
            
            yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)
            end_point = Point(
                x=pose.position.x + 0.2 * math.cos(yaw),
                y=pose.position.y + 0.2 * math.sin(yaw),
                z=0.1
            )
            particles_marker.points.append(end_point)
        
        marker_array.markers.append(particles_marker)
        
        # Create marker for best particle
        best_marker = Marker()
        best_marker.header = msg.header
        best_marker.ns = 'best_particle'
        best_marker.id = 2
        best_marker.type = Marker.ARROW
        best_marker.action = Marker.ADD
        best_marker.scale = self.best_arrow_scale
        best_marker.color = self.best_particle_color
        best_marker.lifetime.sec = 0
        best_marker.lifetime.nanosec = int(self.marker_lifetime * 1e9)
        
        # Add arrow for best particle
        start_point = Point(x=best_pose.position.x, y=best_pose.position.y, z=0.1)
        best_marker.points.append(start_point)
        
        yaw = 2.0 * math.atan2(best_pose.orientation.z, best_pose.orientation.w)
        end_point = Point(
            x=best_pose.position.x + 0.6 * math.cos(yaw),
            y=best_pose.position.y + 0.6 * math.sin(yaw),
            z=0.1
        )
        best_marker.points.append(end_point)
        
        marker_array.markers.append(best_marker)
        
        # Add variance visualization
        variance = np.var(positions, axis=0)
        variance_marker = self.create_variance_marker(
            mean_pos,
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