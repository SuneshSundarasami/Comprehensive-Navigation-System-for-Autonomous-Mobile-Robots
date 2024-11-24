

import rclpy
from rclpy.node import Node
import time
import math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import tf_transformations
from geometry_msgs.msg import Twist
import numpy as np

class PotentialFieldMappingModel(Node):
    def __init__(self):
        super().__init__('PotentialFieldMappingModel_node')

        self.target={'x':1.5,
                     'y':-3.5,
                     'theta':2}
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',  
            self.odom_callback,
            10)
        
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',  
            self.scan_callback,
            10)
        
        self.__goal={
            "x":4.0,
            "y":10.0,
            "theta":-1.0
        }

        self.__ka= 1


    def odom_callback(self, msg):
        # self.get_logger().info(f"-------------------------------------------------------------------")
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        current_z = msg.pose.pose.position.z
        current_orientation = msg.pose.pose.orientation

        # self.get_logger().info(f"Robot Position: x={current_x}, y={current_y}, z={current_z}")
        # self.get_logger().info(f"Robot Orientation: x={current_orientation.x}, y={current_orientation.y}, z={current_orientation.z}, w={current_orientation.w}")

        ai, aj, ak=tf_transformations.euler_from_quaternion([current_orientation.x,current_orientation.y
                                         ,current_orientation.z,current_orientation.w])
        
        self.get_logger().info(f"Robot Position: x={current_x}, y={current_y}, theta={ak}")
        self.get_logger().info(f"Robot Orientation: row={ai}, pitch={aj}, yaw={ak}")

        current_position=np.array([current_x,current_y])

        goal_position= np.array([self.__goal['x'],self.__goal['y']])


        v_attraction= - (self.__ka)*  (current_position-goal_position) /  np.linalg.norm(current_position-goal_position)


        self.get_logger().info(f"Attraction velocities: v_attraction:{ v_attraction} v_attraction_x={v_attraction[0]}, v_attraction_y={v_attraction[1]}")



        twist=Twist()
        twist.linear.x=v_attraction[0]
        twist.linear.y=v_attraction[1]
        self.publisher.publish(twist)


    def scan_callback(self, msg):
        angleArr = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)
        self.get_logger().info(f"size of angle Arr: {angleArr.shape[0]}")
        self.get_logger().info(f"ranges -> { ranges}")
        self.get_logger().info(f"ranges -> { ranges.shape[0]} type - >{ type(ranges)}")


def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldMappingModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
