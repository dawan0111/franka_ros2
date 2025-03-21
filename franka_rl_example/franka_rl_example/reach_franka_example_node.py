import rclpy
import random
import numpy as np
import onnxruntime as ort
import time
from collections import deque
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import os

from ament_index_python.packages import get_package_share_directory 


def get_tf_mat(i, dh):
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    q = theta

    return np.array([[np.cos(q), -np.sin(q), 0, a],
                     [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                     [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                     [0, 0, 0, 1]])


def get_fk_solution(joint_angles):
    dh_params = [[0, 0.333, 0, joint_angles[0]],
                 [0, 0, -np.pi/2, joint_angles[1]],
                 [0, 0.316, np.pi/2, joint_angles[2]],
                 [0.0825, 0, np.pi/2, joint_angles[3]],
                 [-0.0825, 0.384, -np.pi/2, joint_angles[4]],
                 [0, 0, np.pi/2, joint_angles[5]],
                 [0.088, 0, np.pi/2, joint_angles[6]],
                 [0, 0.107, 0, 0],
                 [0, 0, 0, -np.pi/4],
                 [0.0, 0.1034, 0, 0]]

    T = np.eye(4)
    for i in range(7 + 3):
        T = T @ get_tf_mat(i, dh_params)
    return T

class MovingAverageFilter:
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, new_input):
        self.buffer.append(new_input)
        if len(self.buffer) == self.window_size:
            return np.mean(np.array(self.buffer), axis=0)
        else:
            return None
        
    def reset(self):
        self.buffer.clear()

class ReachFrankaExampleNode(Node):
    def __init__(self):
        super().__init__('reach_franka_example_node')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/joint_position_example_controller/commands', 10)
        self.pose_publisher = self.create_publisher(PoseStamped, '/pose_command_marker', 10)
        self.robot_state_subscriber = self.create_subscription(JointState, '/joint_states', self.robot_state_callback, 10)
        # control frequency is 20Hz
        self.timer = self.create_timer(0.02, self.loop)
        self.get_logger().info("reach_franka_example_node has been started!")

        package_share_dir = get_package_share_directory('franka_rl_example')
        self.model_path = os.path.join(package_share_dir, 'models', 'reach.onnx')
        
        self.get_logger().info(f"Loading ONNX model from: {self.model_path}")
        self.ort_session = ort.InferenceSession(self.model_path)
        self.get_logger().info("reach_franka_example_node has been started!")

        self.joint_pos_dim = 9
        self.joint_vel_dim = 9
        self.pose_command_dim = 7 # x, y, z, (in quaternion) x, y, z, w
        self.action_dim = 7
        self.obs_dim = self.joint_pos_dim + self.joint_vel_dim + self.pose_command_dim + self.action_dim

        self.obs = np.zeros(self.obs_dim)
        self.joint_pos = np.zeros(self.joint_pos_dim, dtype=np.float32)
        self.joint_vel = np.zeros(self.joint_vel_dim, dtype=np.float32)
        self.pose_command = np.zeros(self.pose_command_dim, dtype=np.float32)
        self.action = np.zeros(self.action_dim, dtype=np.float32)

        self.default_pos = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
        self.default_vel = np.array([0, 0, 0, 0, 0, 0, 0])

        self.output_scale = 0.5

        self.initialize = False
        self.done = True
        self.tick = 0
        self.filter = MovingAverageFilter(window_size=100)
        self.start_time = time.time()
        self.robot_joint_pos = np.array(self.joint_pos_dim)
        
    def loop(self):
        if not self.initialize:
            return

        self.done_check()

        if self.done:
            self.pose_command = np.array(self.generate_pose_command())
            self.done = False
            self.start_time = time.time()
            self.filter.reset()
            return

        if self.tick <= 250:
            msg = Float64MultiArray()
            msg.data = self.default_pos.tolist()
            self.publisher_.publish(msg)
            self.publish_pose_marker()
            self.tick += 1
            return
        

        ee_pose = get_fk_solution(self.robot_joint_pos[:7].tolist())
        ee_trans = ee_pose[:3, 3]
        goal_trans = self.pose_command[:3]

        # print(ee_trans, goal_trans)
        dist = np.linalg.norm(ee_trans - goal_trans)

        # if dist <= 0.7:
        #     msg = Float64MultiArray()
        #     msg.data = self.robot_joint_pos[:7].tolist()
        #     self.publisher_.publish(msg)
        #     return
        
        input_name = self.ort_session.get_inputs()[0].name
    
        output = self.ort_session.run(None, {input_name: self.get_input_vector()})[0]
        output = output.reshape(-1)
        msg = Float64MultiArray()
        for i in range(len(output)):
            # print(self.default_pos[i], output[i])
            msg.data.append(self.default_pos[i] + output[i] * self.output_scale)

        filtered_output = self.filter.update(np.array(msg.data))

        if filtered_output is not None:
            msg.data = filtered_output.tolist()

        self.publisher_.publish(msg)
        self.publish_pose_marker()

        self.action = output.reshape(-1)
        # print(self.action)
        # self.get_logger().info(f'Publishing joint positions: {msg.data}')

    def done_check(self):
        if time.time() - self.start_time > 5:
            self.done = True
            self.get_logger().info("Done")
            return True
        
        return False

    def publish_pose_marker(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "world"

        pose_msg.pose.position.x = self.pose_command[0]
        pose_msg.pose.position.y = self.pose_command[1]
        pose_msg.pose.position.z = self.pose_command[2]

        pose_msg.pose.orientation.x = self.pose_command[4]
        pose_msg.pose.orientation.y = self.pose_command[5]
        pose_msg.pose.orientation.z = self.pose_command[6]
        pose_msg.pose.orientation.w = self.pose_command[3]

        self.pose_publisher.publish(pose_msg)

    def get_input_vector(self):
        return np.concatenate([self.joint_pos, self.joint_vel, self.pose_command, self.action]).reshape(1, -1).astype(np.float32)

    def generate_pose_command(self):
      # XYZ 위치 (정규분포 기반 무작위 값)
      x = random.uniform(0.35, 0.65)
      y = random.uniform(-0.2, 0.2)
      z = random.uniform(0.15, 0.5)

      # Yaw 회전 (기본 방향을 유지하면서 yaw만 적용)
      yaw = random.uniform(-np.pi, np.pi)

      # 🔹 Franka Panda 기본 EE 방향을 quaternion으로 직접 설정
      # 기본적으로 x-축이 전방, z-축이 아래를 향하는 방향
      qx_base, qy_base, qz_base, qw_base = 1.0, 0.0, 0.0, 0.0  # 180도 회전된 기본 방향 (x-forward, z-down)

      # Yaw(ψ) 회전을 quaternion으로 변환
      qx_yaw = 0.0
      qy_yaw = 0.0
      qz_yaw = np.sin(yaw / 2)
      qw_yaw = np.cos(yaw / 2)

      # 두 개의 quaternion을 곱해서 최종 회전 적용
      # (qx_base, qy_base, qz_base, qw_base) * (qx_yaw, qy_yaw, qz_yaw, qw_yaw)
      qx = qw_base * qx_yaw + qx_base * qw_yaw + qy_base * qz_yaw - qz_base * qy_yaw
      qy = qw_base * qy_yaw - qx_base * qz_yaw + qy_base * qw_yaw + qz_base * qx_yaw
      qz = qw_base * qz_yaw + qx_base * qy_yaw - qy_base * qx_yaw + qz_base * qw_yaw
      qw = qw_base * qw_yaw - qx_base * qx_yaw - qy_base * qy_yaw - qz_base * qz_yaw

      return x, y, z, qw, qx, qy, qz

    def robot_state_callback(self, msg):
        # self.get_logger().info(msg.position)
        # why 1 -> 3 -> 2 ??
        msg.position[2], msg.position[1] = msg.position[1], msg.position[2]
        msg.velocity[2], msg.velocity[1] = msg.velocity[1], msg.velocity[2]

        for i in range(len(msg.position)):
            self.joint_pos[i] = msg.position[i] - self.default_pos[i]
            self.joint_vel[i] = msg.velocity[i] - self.default_vel[i] 

        self.joint_pos[-1] = -0.004
        self.joint_pos[-2] = -0.004

        self.joint_vel[-1] = -0.004
        self.joint_vel[-2] = -0.004

        self.initialize = True
        self.robot_joint_pos = np.array(msg.position)

        # print("pos:", self.joint_pos)
        # print("vel:", self.joint_vel)

def main(args=None):
    rclpy.init(args=args)
    node = ReachFrankaExampleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()