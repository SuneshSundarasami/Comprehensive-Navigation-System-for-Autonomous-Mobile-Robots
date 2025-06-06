import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Pose2D
from nav_msgs.msg import Path
from std_msgs.msg import String
from std_srvs.srv import Trigger
import tf_transformations
import time

class PoseExecutor(Node):
    def __init__(self):
        super().__init__('pose_executor')
        
        # Add progress publisher
        self.progress_pub = self.create_publisher(
            String,
            '/pose_progress',
            10
        )
        
        # Remove status subscription and create service client
        self.status_client = self.create_client(Trigger, 'get_pfield_status')
        
        # Wait for service to become available
        while not self.status_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Status service not available, waiting...')
            
        self.status_request = Trigger.Request()
        
        # Create timer for periodic status checking
        self.create_timer(0.5, self.check_status)  # Check status every 0.5 seconds
        
        self.pose_sub = self.create_subscription(
            Path, 
            '/planned_path', 
            self.path_callback, 
            10
        )
        
        self.pose_pub = self.create_publisher(
            Pose2D, 
            '/end_pose', 
            10
        )
        
        self.pose_list = []
        self.current_index = 0
        self.executing = False
        self.waiting_for_trigger = False
        self.last_status = ""

    async def get_status(self):
        """Asynchronously get the status from the service"""
        future = self.status_client.call_async(self.status_request)
        await future
        return future.result()

    def check_status(self):
        """Periodic status check using the service"""
        if not self.executing:
            return
            
        # Create a new future for the service call
        future = self.status_client.call_async(self.status_request)
        future.add_done_callback(self.status_callback)

    def status_callback(self, future):
        """Callback for handling the service response"""
        try:
            response = future.result()
            status = response.message
            
            # Only process if status has changed
            if status != self.last_status:
                self.last_status = status
                if status in ['Goal Position Reached! Alligned orientation!', 'Waiting for goal pose']:
                    progress_msg = f"Completed pose {self.current_index + 1}/{len(self.pose_list)}"
                    if self.current_index + 1==len(self.pose_list):
                        progress_msg="All poses completed!"
                    self.publish_progress(progress_msg)
                    
                    time.sleep(2)
                    self.publish_next_pose()
                    
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    def path_callback(self, msg):
        """Receives the planned path and updates execution."""
        new_poses = msg.poses
        if not new_poses:
            return
            
        # Check if path has changed
        if self.has_path_changed(new_poses):
            self.get_logger().info("New path detected, updating poses...")
            self.pose_list = new_poses
            self.current_index = 0
            self.executing = True
            self.waiting_for_trigger = False
            
            # Publish initial path information
            progress_msg = f"Starting execution of {len(self.pose_list)} poses"
            self.publish_progress(progress_msg)
            
            # Print all received poses
            print("Received New Poses:")
            for i, pose in enumerate(self.pose_list):
                x, y, theta = self.extract_pose_2d(pose)
                print(f"Pose {i+1}: x={x:.2f}, y={y:.2f}, theta={theta:.2f}")
            self.publish_current_pose()

    def has_path_changed(self, new_poses):
        """Check if the new path is different from current path."""
        if len(new_poses) != len(self.pose_list):
            return True
        
        for new_pose, old_pose in zip(new_poses, self.pose_list):
            new_x, new_y, new_theta = self.extract_pose_2d(new_pose)
            old_x, old_y, old_theta = self.extract_pose_2d(old_pose)
            
            # Check if positions differ by more than a small threshold
            if (abs(new_x - old_x) > 0.01 or 
                abs(new_y - old_y) > 0.01 or 
                abs(new_theta - old_theta) > 0.01):
                return True
        
        return False

    def extract_pose_2d(self, pose):
        """Extracts x, y, and yaw from a PoseStamped message."""
        x = pose.pose.position.x
        y = pose.pose.position.y
        quaternion = (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w
        )
        euler = tf_transformations.euler_from_quaternion(quaternion)
        theta = euler[2]
        return x, y, theta

    def publish_current_pose(self):
        """Publishes the current pose and prints it."""
        if self.current_index < len(self.pose_list):
            x, y, theta = self.extract_pose_2d(self.pose_list[self.current_index])
            
            # Print only the current pose being executed
            print(f"Executing Pose {self.current_index + 1}: x={x}, y={y}, theta={theta}")
            
            pose2d = Pose2D()
            pose2d.x = x
            pose2d.y = y
            pose2d.theta = theta
            self.pose_pub.publish(pose2d)
            
            if not self.waiting_for_trigger:
                self.waiting_for_trigger = True
                self.create_timer(2.0, self.publish_current_pose)

    def publish_next_pose(self):
        """Publishes the next pose in the list."""
        if self.current_index >= len(self.pose_list):
            self.executing = False
            self.publish_progress("All poses completed!")
            return
            
        self.waiting_for_trigger = False
        self.current_index += 1
        self.publish_current_pose()

    def publish_progress(self, message):
        """Publish progress updates"""
        msg = String()
        msg.data = message
        self.progress_pub.publish(msg)
        self.get_logger().info(message)

def main(args=None):
    rclpy.init(args=args)
    node = PoseExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()