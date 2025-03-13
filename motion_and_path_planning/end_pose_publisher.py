import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D

class EndPosePublisher(Node):
    def __init__(self):
        super().__init__('endpose_publisher')
        self.publisher_ = self.create_publisher(Pose2D, 'goal_pose', 10)
        self.timer = self.create_timer(1/50.0, self.publish_end_pose)  # Publish at 1 Hz

    def publish_end_pose(self):
        end_pose = Pose2D()
        end_pose.x = 6.0   
        end_pose.y = 0.0   
        end_pose.theta = 1.57 

        self.publisher_.publish(end_pose)
        self.get_logger().info(f'Published End Pose: x={end_pose.x}, y={end_pose.y}, theta={end_pose.theta}')

def main(args=None):
    rclpy.init(args=args)
    node = EndPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
