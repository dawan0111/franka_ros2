import rclpy
import numpy as np
import onnxruntime as ort
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import os
from ament_index_python.packages import get_package_share_directory 

class ReachFrankaExampleNode(Node):
    def __init__(self):
        super().__init__('reach_franka_example_node')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/joint_position_example_controller/commands', 10)
        # control frequency is 20Hz
        self.timer = self.create_timer(0.05, self.publish_joint_positions)
        self.get_logger().info("reach_franka_example_node has been started!")

        package_share_dir = get_package_share_directory('franka_rl_example')
        self.model_path = os.path.join(package_share_dir, 'models', 'reach.onnx')
        
        self.get_logger().info(f"Loading ONNX model from: {self.model_path}")
        self.ort_session = ort.InferenceSession(self.model_path)
        self.get_logger().info("reach_franka_example_node has been started!")

        self.done = False
        

    def publish_joint_positions(self):
        msg = Float64MultiArray()
        msg.data = [1.57, 0.0, -0.7853981634, -2.3561944902, 0.0, 1.5707963268, 0.7853981634]
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing joint positions: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = ReachFrankaExampleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()