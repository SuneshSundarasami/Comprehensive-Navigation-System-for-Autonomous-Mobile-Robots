import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
import time

class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_publisher')
        self.publisher = self.create_publisher(Pose2D, 'end_pose', 10)
        
    def publish_test_goal(self):
        msg = Pose2D()
        msg.x = 1.0  # 1 meter forward
        msg.y = 0.0
        msg.theta = 0.0
        self.publisher.publish(msg)
        self.get_logger().info('Published test goal')

def main(args=None):
    rclpy.init(args=args)
    node = TestPublisher()
    time.sleep(2)  # Wait for connections
    node.publish_test_goal()
    time.sleep(1)
    rclpy.shutdown()

if __name__ == '__main__':
    main()