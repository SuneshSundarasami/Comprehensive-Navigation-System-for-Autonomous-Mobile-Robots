

import rclpy
from rclpy.node import Node
import time
import math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
import tf_transformations
from geometry_msgs.msg import Twist
import numpy as np
import tf2_ros

class PotentialFieldMappingModel(Node):
    def __init__(self):
        super().__init__('PotentialFieldMappingModel_node')

        self.target={'x':1.5,
                     'y':-3.5,
                     'theta':2}
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.odom_subscription = self.create_subscription(
        #     Odometry,
        #     '/odom',  
        #     self.odom_callback,
        #     10)
        
        # self.scan_subscription = self.create_subscription(
        #     LaserScan,
        #     '/scan',  
        #     self.scan_callback,
        #     10)
        
        # self.tf_subscription = self.create_subscription(
        #     TFMessage,  
        #     '/tf_static', 
        #     self.tf_callback,
        #     10  
        # )
        
        self.__goal={
            "x":4.0,
            "y":10.0,
            "theta":-1.0
        }

        self.__ka= 1


    #     self.tf_buffer = tf2_ros.Buffer()
    #     self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    #     self.timer = self.create_timer(5.0, self.list_transforms)  # Every 5 seconds

    # def list_transforms(self):
    #     try:
    #         # Get all transformations as a string
    #         transforms_str = self.tf_buffer.all_frames_as_string()
    #         if transforms_str:
    #             self.get_logger().info(f"Available Transformations:\n{transforms_str}")
    #         else:
    #             self.get_logger().info("No transformations available.")
    #     except Exception as e:
    #         self.get_logger().error(f"Error listing transformations: {e}")



        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_timer(0.01, self.retrieve_transform)
        self.transform_matrix = None

    def retrieve_transform(self):
        try:
            transform = self.tf_buffer.lookup_transform('odom', 'base_laser_front_link', rclpy.time.Time())
            t, q = transform.transform.translation, transform.transform.rotation

            # Build 4x4 transformation matrix
            self.transform_matrix = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            self.transform_matrix[0:3, 3] = [t.x, t.y, t.z]

            self.get_logger().info(f"Transformation Matrix:\n{self.transform_matrix}")
        except Exception as e:
            self.get_logger().error(f"Transform error: {e}")




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
        # self.publisher.publish(twist)


    def scan_callback(self, msg):
        angleArr = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)
        self.get_logger().info(f"size of angle Arr: {angleArr.shape[0]}")
        self.get_logger().info(f"ranges -> { ranges.shape[0]} type - >{ type(ranges)}")

        np_polar=np.stack([angleArr,ranges],axis=1)



        # below code to be defined in func np_polar2cart - converts polar to cartesian
        cart_arr = np.array([[np.cos(i[0])*i[1], np.sin(i[0])*i[1]] for i in np_polar])

        self.get_logger().info(f"ranges -> { ranges.shape[0]} type - >{ type(ranges)}")

        print(cart_arr)


    def tf_callback(self, msg):
        for i in msg.transforms:
            if i.header.frame_id=='base_link':
                if i.child_frame_id=='base_laser_front_link':
                    print("\n tf static -> ",i)
                    # assign the transforms object of base_link and base_laser_front_link to tf_static for further processing
                    tf_static=i


        



def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldMappingModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
