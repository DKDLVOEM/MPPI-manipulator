#!/usr/bin/env python3

import argparse
import sys
import numpy as np
import torch
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker # <--- 이 줄을 추가하세요.


# 원본 realsense_mpc.py와 동일한 import 구문을 사용합니다.
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.types.robot import JointState as CuroboJointState
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.rollout.rollout_base import Goal


class CuroboMpcNode(Node):
    def __init__(self, args):
        super().__init__('curobo_mpc_node')
        self.args = args
        
        # ROS 관련 객체 초기화
        self.cv_bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 데이터 수신용 변수
        self.depth_image = None
        self.color_image = None
        self.camera_intrinsics = None
        self.latest_joint_state_msg = None

        # Curobo 초기화
        self._setup_curobo()

        # ROS 통신 설정
        self._setup_ros_communications()

        self.get_logger().info("CuRobo MPC Node initialized successfully.")

    def _setup_curobo(self):
        """원본 realsense_mpc.py의 Curobo 설정 로직을 그대로 따릅니다."""
        self.get_logger().info(f"Loading robot configuration: {self.args.robot}")
        
        self.tensor_args = TensorDeviceType()

        # 원본과 동일하게, 코드 내에서 월드를 직접 생성합니다.
        self.world_cfg = WorldConfig.from_dict(
            {
                "blox": {
                    "world": {
                        "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], # float으로 명시
                        "integrator_type": "occupancy",
                        "voxel_size": 0.03,
                    }
                }
            }
        )
        
        # 로봇 설정 로드 (딕셔너리 형태)
        robot_cfg_dict = load_yaml(join_path(get_robot_configs_path(), self.args.robot))["robot_cfg"]
        
        # MPC 설정
        self.step_dt = 0.02
        self.mpc_config = MpcSolverConfig.load_from_robot_config(
            robot_cfg_dict, # 딕셔너리를 직접 전달
            self.world_cfg,
            use_cuda_graph=True,
            use_cuda_graph_metrics=True,
            use_cuda_graph_full_step=False,
            self_collision_check=True,
            collision_checker_type=CollisionCheckerType.BLOX,
            use_mppi=True,
            use_lbfgs=False,
            use_es=False,
            store_rollouts=True,
            step_dt=self.step_dt,
        )

        self.mpc = MpcSolver(self.mpc_config)
        self.world_model = self.mpc.world_collision
        
        # 초기 상태 및 목표 설정
        self.joint_names = self.mpc.rollout_fn.joint_names
        retract_cfg = self.mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
        self.current_state = CuroboJointState.from_position(retract_cfg, joint_names=self.joint_names)
        
        state = self.mpc.rollout_fn.compute_kinematics(self.current_state)
        retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
        
        goal = Goal(
            current_state=self.current_state,
            goal_state=CuroboJointState.from_position(retract_cfg, joint_names=self.joint_names),
            goal_pose=retract_pose,
        )
        goal_buffer = self.mpc.setup_solve_single(goal, 1)
        self.mpc.update_goal(goal_buffer)

        # MPC 루프 제어용 변수
        self.cmd_state_full = None

        self.get_logger().info("CuRobo setup is complete.")

    def _setup_ros_communications(self):
        """ROS2 구독자, 발행자, 타이머 설정"""
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        # 올바른 토픽 이름으로 수정
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, qos)
        self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, qos)
        self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.cam_info_callback, qos)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.traj_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)
        self.camera_marker_pub = self.create_publisher(Marker, '/camera_pose', 10)
        self.create_timer(1.0 / 30.0, self.realsense_callback) # 30Hz
        self.create_timer(1.0 / 50.0, self.mpc_callback)      # 50Hz

    def depth_callback(self, msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
        print("depth callback")
    
    def color_callback(self, msg):
        self.color_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    
    def cam_info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {"width": msg.width, "height": msg.height, "fx": msg.k[0], "fy": msg.k[4], "cx": msg.k[2], "cy": msg.k[5]}
            self.get_logger().info("Camera intrinsics received.")
    
    def joint_state_callback(self, msg):
        # print("joint state callback")

        self.latest_joint_state_msg = msg

    def realsense_callback(self):
        """30Hz: NVBlox 월드 업데이트"""
        if self.depth_image is None or self.camera_intrinsics is None:
            return
            
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'camera_depth_optical_frame', rclpy.time.Time())
            t = trans.transform.translation
            q = trans.transform.rotation
            camera_pose = Pose(
                position=self.tensor_args.to_device(torch.tensor([[t.x, t.y, t.z]])),
                quaternion=self.tensor_args.to_device(torch.tensor([[q.x, q.y, q.z, q.w]])),
            )
            self._publish_camera_marker(camera_pose)
            print("pose marker published")

        except Exception as e:
            self.get_logger().warn(f"TF lookup for camera failed: {e}", throttle_duration_sec=2.0)
            return

        self.world_model.decay_layer("world")
        depth_tensor = self.tensor_args.to_device(torch.from_numpy(self.depth_image))
        
        # nvblox에 전달하기 전, 딕셔너리 내부의 값들을 텐서로 변환
        intrinsics_tensor_dict = {k: torch.as_tensor(v, device=self.tensor_args.device) for k, v in self.camera_intrinsics.items()}
        data_camera = CameraObservation(depth_image=depth_tensor, intrinsics=intrinsics_tensor_dict, pose=camera_pose)
        
        self.world_model.add_camera_frame(data_camera, "world")
        self.world_model.process_camera_frames("world", False)
        torch.cuda.synchronize()

        if self.args.show_window and self.color_image is not None:
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((self.color_image, depth_colormap))
            cv2.imshow("RealSense View (ROS)", images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Shutdown requested via CV window.")
                rclpy.shutdown()

    def mpc_callback(self):
        """50Hz: MPC 계산 및 제어 명령 발행"""
        if self.latest_joint_state_msg is None:
            return
            
        try:
            # 현재 로봇 상태 업데이트
            ordered_positions = []
            for name in self.joint_names:
                idx = self.latest_joint_state_msg.name.index(name)
                ordered_positions.append(self.latest_joint_state_msg.position[idx])
            
            cu_js = CuroboJointState.from_position(
                self.tensor_args.to_device(torch.tensor([ordered_positions]))
            )
            
            # 원본 코드의 상태 업데이트 로직을 유지
            if self.cmd_state_full is None:
                self.current_state.copy_(cu_js)
            else:
                self.current_state.copy_(self.cmd_state_full.get_ordered_joint_state(self.joint_names))

        except ValueError:
            self.get_logger().warn("Joint name mismatch in /joint_states.", throttle_duration_sec=5.0)
            return
        
        # MPC 스텝 실행
        mpc_result = self.mpc.step(self.current_state, max_attempts=2)

        if 1:
        # if mpc_result.success.item():
            self.cmd_state_full = mpc_result.js_action
            
            traj_msg = JointTrajectory()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.joint_names = self.joint_names
            
            point = JointTrajectoryPoint()
            ordered_cmd = self.cmd_state_full.get_ordered_joint_state(self.joint_names)
            point.positions = ordered_cmd.position.cpu().numpy().flatten().tolist()
            point.velocities = ordered_cmd.velocity.cpu().numpy().flatten().tolist()
            point.time_from_start = Duration(sec=0, nanosec=int(self.step_dt * 1e9))
            
            traj_msg.points.append(point)
            self.traj_pub.publish(traj_msg)
        else:
            self.get_logger().warn("MPC step failed.", throttle_duration_sec=1.0)


    def _publish_camera_marker(self, pose: Pose):
        """
        주어진 Pose 정보를 RViz에서 볼 수 있는 Marker로 변환하여 발행합니다.
        카메라의 위치와 방향을 화살표(Arrow) 형태로 시각화합니다.
        """
        marker = Marker()
        marker.header.frame_id = self.base_link # Marker는 base_link를 기준으로 표시됩니다.
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "camera_pose"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Pose 객체의 텐서 데이터를 ROS 메시지 타입으로 변환합니다.
        position = pose.position.cpu().numpy().flatten()
        quat = pose.quaternion.cpu().numpy().flatten()
        
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.x = float(quat[0])
        marker.pose.orientation.y = float(quat[1])
        marker.pose.orientation.z = float(quat[2])
        marker.pose.orientation.w = float(quat[3])

        # 화살표의 크기 설정
        marker.scale.x = 0.2  # 화살표 길이
        marker.scale.y = 0.02 # 화살표 두께
        marker.scale.z = 0.02 # 화살표 머리 두께

        # 화살표 색상 설정 (빨간색)
        marker.color.a = 1.0  # Alpha (불투명도)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        # 발행
        self.camera_marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="ur5e.yml", help="robot configuration to load")
    parser.add_argument("--show-window", action="store_true", default=False, help="When True, shows camera image in a CV window")
    
    ros_args_only = rclpy.utilities.remove_ros_args(args=sys.argv)
    args = parser.parse_args(ros_args_only[1:])
    
    node = CuroboMpcNode(args)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if args.show_window:
            cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()