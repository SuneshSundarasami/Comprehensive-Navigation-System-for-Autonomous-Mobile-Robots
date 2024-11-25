

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
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_timer(0.01, self.retrieve_transform)
        self.transform_matrix = np.eye(4)

    def retrieve_transform(self):
        try:
            transform = self.tf_buffer.lookup_transform('odom', 'base_laser_front_link', rclpy.time.Time())
            t, q = transform.transform.translation, transform.transform.rotation

            # Build 4x4 transformation matrix
            self.transform_matrix = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            self.transform_matrix[0:3, 3] = [t.x, t.y, t.z]

            # self.get_logger().info(f"Transformation Matrix:\n{self.transform_matrix}")
        except Exception as e:
            self.get_logger().error(f"Transform error: {e}")



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

        self.__ka= 0.5

        self.__kr= 1.5

        self.__distance_threshold= 2

        self.current_position=np.array([np.inf,np.inf])

        self.v_attraction= np.zeros((2,))

        self.v_repulsion= np.zeros((2,))






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
        # self.get_logger().info(f"Robot Orientation: row={ai}, pitch={aj}, yaw={ak}")

        self.current_position=np.array([current_x,current_y])

        goal_position= np.array([self.__goal['x'],self.__goal['y']])


        self.v_attraction= - (self.__ka)*  (self.current_position-goal_position) /  np.linalg.norm(self.current_position-goal_position)


        self.get_logger().info(f"Attraction velocities: v_attraction_x={self.v_attraction[0]}, v_attraction_y={self.v_attraction[1]}")



        twist=Twist()
        twist.linear.x=self.v_attraction[0]
        twist.linear.y=self.v_attraction[1]
        # self.publisher.publish(twist)


    def scan_callback(self, msg):
        angleArr = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)
        # self.get_logger().info(f"size of angle Arr: {angleArr.shape[0]}")
        # self.get_logger().info(f"ranges -> { ranges.shape[0]} type - >{ type(ranges)}")

        np_polar=np.stack([angleArr,ranges],axis=1)



        # below code to be defined in func np_polar2cart - converts polar to cartesian
        cart_arr = np.array([[np.cos(i[0])*i[1], np.sin(i[0])*i[1]] for i in np_polar])

        # self.get_logger().info(f"ranges -> { ranges.shape[0]} type - >{ type(ranges)}")


        cart_arr_with_z = np.hstack((cart_arr,np.zeros((cart_arr.shape[0],1)), np.ones((cart_arr.shape[0],1))))
        cart_arr_transpose = np.transpose(cart_arr_with_z)
        #new_rotation_matrix = Rotational matrix * laser scan data
        tranformed_arr = (self.transform_matrix  @ cart_arr_transpose).T

        # Removed inf points
        obst_coords = tranformed_arr[~np.isinf(tranformed_arr).any(axis=1)]

        obst_coords=np.array(obst_coords[:,:2])


        # self.get_logger().info(f"Obstacle Co-ordinates(odom) -> { obst_coords} ")

        self.v_repulsion= np.zeros((2,))

        for obst_coord in obst_coords:

            if np.linalg.norm(self.current_position-obst_coord)<self.__distance_threshold:

                v_repulsion_i=  (self.__kr)*  ( (1 /  np.linalg.norm(self.current_position-obst_coord))- (1/self.__distance_threshold)) * ((self.current_position-obst_coord)/ ((np.linalg.norm(self.current_position-obst_coord))**3) )
                # self.get_logger().info(f"Replusive velocities -> { v_repulsion_i} ")

                self.v_repulsion+=v_repulsion_i
                # self.get_logger().info(f"Replusive velocities(total) -> { self.v_repulsion} ")

        self.get_logger().info(f"Replusion velocities: v_repulsion_x={self.v_repulsion[0]}, v_repulsion_y={self.v_repulsion[1]}")
        twist=Twist()
        if (self.v_repulsion[0]==np.nan) or (self.v_repulsion[1]==np.nan):
            self.v_repulsion=np.zeros((2,))
        twist.linear.x=self.v_repulsion[0]
        twist.linear.y=self.v_repulsion[1]

        self.v_total=self.v_attraction+ self.v_repulsion
        # self.v_total=self.v_attraction

        self.get_logger().info(f"Final velocity Update: v_total_x={self.v_total[0]}, v_total_y={self.v_total[1]}")




        twist=Twist()
        twist.linear.x=self.v_total[0]
        twist.linear.y=self.v_total[1]
        self.publisher.publish(twist)




        



def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldMappingModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
