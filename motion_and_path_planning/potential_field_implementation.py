

import rclpy
from rclpy.node import Node
import time
import math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
import tf_transformations
from geometry_msgs.msg import Twist
import numpy as np
import tf2_ros
from geometry_msgs.msg import Pose2D
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_srvs.srv import Trigger

class PotentialFieldMappingModel(Node):
    def __init__(self):
        super().__init__('PotentialFieldMappingModel_node')
        qos_best_effort = QoSProfile(
            depth=1, 
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        self.current_orientation=None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_timer(0.001, self.retrieve_transform)
        self.status_service = self.create_service(
            Trigger,
            'get_pfield_status',
            self.get_status_callback
        )
        self.pfield_status='Not Started Yet!'

        self.__goal={
            "x":np.nan,
            "y":np.nan,
            "theta":np.nan
        }

        

    
    
    def retrieve_transform(self):
        try:
            transform = self.tf_buffer.lookup_transform('odom', 'base_laser_front_link', rclpy.time.Time())
            t, q = transform.transform.translation, transform.transform.rotation

            # Build 4x4 transformation matrix
            self.transform_matrix = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            self.transform_matrix[0:3, 3] = [t.x, t.y, t.z]


        except Exception as e:
            self.get_logger().error(f"Transform error: {e}")

        self.transform_matrix = np.eye(4)
        self.log_counter=0
        self.goal_subscription = self.create_subscription(
            Pose2D,
            'end_pose',  # Topic name
            self.goal_callback,
            10
        )

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
        

        
        

        self.__ka= 1

        self.__kr= 0.2

        self.__distance_threshold= 1.0

        self.current_position=np.array([np.inf,np.inf])

        self.v_attraction= np.zeros((2,))

        self.v_repulsion= np.zeros((2,))

        self.__goal_pose_error=0.2


   


    def get_status_callback(self, request, response):
        """Callback for the status service"""
        response.success = True
        response.message = self.pfield_status
        return response


    def goal_callback(self, msg):
        self.__goal["x"] = msg.x
        self.__goal["y"] = msg.y
        self.__goal["theta"] = msg.theta
        self.pfield_status="Moving to Goal Pose!"
        self.log_now=self.log_counter%200==0
        if self.log_now:
            self.get_logger().info(f'Updated Goal: x={msg.x}, y={msg.y}, theta={msg.theta}')
        


    def odom_callback(self, msg):
        # self.get_logger().info(f"-------------------------------------------------------------------")
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        current_z = msg.pose.pose.position.z
        current_orientation = msg.pose.pose.orientation

        self.current_orientation=current_orientation

        # self.get_logger().info(f"Robot Position: {self.current_orientation}")

        # self.get_logger().info(f"Robot Position: x={current_x}, y={current_y}, z={current_z}")
        # self.get_logger().info(f"Robot Orientation: x={current_orientation.x}, y={current_orientation.y}, z={current_orientation.z}, w={current_orientation.w}")

        ai, aj, ak=tf_transformations.euler_from_quaternion([current_orientation.x,current_orientation.y
                                         ,current_orientation.z,current_orientation.w])
        
        # self.get_logger().info(f"Robot Position: x={current_x}, y={current_y}, theta={ak}")
        # self.get_logger().info(f"Robot Orientation: row={ai}, pitch={aj}, yaw={ak}")

        self.current_position=np.array([current_x,current_y])

        self.goal_position= np.array([self.__goal['x'],self.__goal['y']])

        if np.isnan(self.__goal['x']):
            self.get_logger().info(f"Waiting for goal pose")
            return

        self.log_now=self.log_counter%20==0
        if self.log_now:
            self.get_logger().info(f"distance to the goal ----------------------------------------------{np.linalg.norm(self.current_position-self.goal_position)}->>>>>>>")

        if np.linalg.norm(self.current_position-self.goal_position)< self.__goal_pose_error:
            self.goal_allign(ak)
        else:
            self.pfield_status="Moving to Goal Pose!"
            self.v_attraction= - (self.__ka)*  (self.current_position-self.goal_position) /  np.linalg.norm(self.current_position-self.goal_position)


            # self.get_logger().info(f"Attraction velocities: v_attraction_x={self.v_attraction[0]}, v_attraction_y={self.v_attraction[1]}")



            twist=Twist()
            twist.linear.x=self.v_attraction[0]
            twist.linear.y=self.v_attraction[1]
            # self.publisher.publish(twist)


    def scan_callback(self, msg):
        if np.isnan(self.__goal['x']):
            return
        
        # print(angleArr.shape,ranges.shape)
        angleArr = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)
        ranges=ranges[:len(angleArr)]
        angleArr=ranges[:len(ranges)]
        # self.get_logger().info(f"size of angle Arr: {angleArr.shape[0]}")
        # self.get_logger().info(f"ranges -> { ranges.shape[0]} type - >{ type(ranges)}")

        # print(angleArr.shape,ranges.shape)
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

    
        twist=Twist()
        if (self.v_repulsion[0]==np.nan) or (self.v_repulsion[1]==np.nan):
            self.v_repulsion=np.zeros((2,))
        twist.linear.x=self.v_repulsion[0]
        twist.linear.y=self.v_repulsion[1]

        self.v_total=self.v_attraction+ self.v_repulsion
        # self.v_total=self.v_attraction

        self.log_counter+=1
        # print(self.log_counter)

        self.log_now=self.log_counter%20==0

        if self.log_now:

            self.get_logger().info(f"Final velocity Update: v_total_x={self.v_total[0]}, v_total_y={self.v_total[1]}")

        v_total_magnitude = np.linalg.norm(self.v_total)  # Magnitude of total velocity
        v_total_angle = math.atan2(self.v_total[1], self.v_total[0])  # Angle of total velocity


        if self.log_now:
            self.get_logger().info(f"Velocity magnitude: {v_total_magnitude}, Velocity angle: {v_total_angle}")
        # self.get_logger().info(f"Robot Position: {self.current_orientation}")
        # Calculate angular velocity (wz) based on heading difference
        current_heading = tf_transformations.euler_from_quaternion([self.current_orientation.x, 
                                                                     self.current_orientation.y, 
                                                                     self.current_orientation.z, 
                                                                     self.current_orientation.w])[2]
        heading_difference = v_total_angle - current_heading

        # Normalize heading difference to [-pi, pi]
        heading_difference = math.atan2(math.sin(heading_difference), math.cos(heading_difference))

        max_vel=0.25
        # Forward velocity and angular velocity

        # twist.linear.x=if twist.linear.x
        if np.linalg.norm(self.current_position-self.goal_position)<0.5:
            max_vel=0.1

        twist = Twist()
        twist.linear.x,twist.angular.z =self.limit_velocities(v_total_magnitude,heading_difference,max_vel)

        # if self.v_total[0]<0:
        #     twist.linear.x,twist.angular.z =-twist.linear.x,-twist.angular.z 
        # twist.linear.x = min(v_total_magnitude, max_vel)  # Cap forward velocity to a maximum of 1.0
        # twist.angular.z = max_vel if heading_difference > max_vel else (-max_vel if heading_difference < -max_vel else heading_difference)

        # min(1 * heading_difference, 0.5)    # Proportional control for angular velocity

        if self.log_now:
            self.get_logger().info(f"Final velocity calculated: linear_x={v_total_magnitude}, angular_z={heading_difference}")
            self.get_logger().info(f"Final velocity applied: linear_x={twist.linear.x}, angular_z={twist.angular.z}")

        
        self.publisher.publish(twist)

    def limit_velocities(self,v_total_magnitude, heading_difference, max_vel):
        # Calculate the ratio of the velocities
        ratio = abs(heading_difference / v_total_magnitude) if v_total_magnitude != 0 else float('inf')
        
        # If either velocity is above max_vel, scale both proportionally
        if v_total_magnitude > max_vel or abs(heading_difference) > max_vel:
            if ratio > 1:  # Angular velocity is larger
                # Scale down angular velocity to max_vel and linear proportionally
                new_angular = max_vel if heading_difference > 0 else -max_vel
                new_linear = v_total_magnitude / abs(heading_difference) * max_vel
            else:  # Linear velocity is larger
                # Scale down linear velocity to max_vel and angular proportionally
                new_linear = max_vel
                new_angular = heading_difference / v_total_magnitude * max_vel
        else:
            # If neither exceeds max_vel, use original values
            new_linear = v_total_magnitude
            new_angular = heading_difference
            
        return new_linear, new_angular

    def goal_allign(self,z_angle):

        self.publisher.publish(Twist())

        

        if np.abs(z_angle- self.__goal['theta'])>self.__goal_pose_error/2:
            self.pfield_status="Goal Position Reached! Alligning orientation!"
            self.get_logger().info(f"Goal Position Reached! Alligning orientation.... | Angle diff:{(z_angle- self.__goal['theta'])}")
            twist=Twist()
            twist.angular.z=0.1 if (z_angle- self.__goal['theta'])<0 else -0.1
            self.publisher.publish(twist)
            time.sleep(0.1)

        else:
            self.pfield_status="Goal Position Reached! Alligned orientation!"
            self.__goal={
            "x":np.nan,
            "y":np.nan,
            "theta":np.nan
        }
            self.get_logger().info(f"Goal to be reached x:{self.__goal['x']}, y:{self.__goal['y']}, theta:{self.__goal['theta']}")
            self.get_logger().info(f"Goal Position Reached! Alligned orientation! x:{self.current_position[0]}, y:{self.current_position[1]}, theta:{z_angle}")
            # self.get_logger().info('Shutting down the node...')
            # rclpy.shutdown()  



def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldMappingModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
