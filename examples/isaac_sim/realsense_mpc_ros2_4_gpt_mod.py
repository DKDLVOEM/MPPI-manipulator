#!/usr/bin/env python3

import argparse
import sys
import numpy as np
import torch
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker

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

from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header


def _quat_mul(q1, q2):
    # q = q1 * q2, 각 quaternion은 (x, y, z, w)
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float32)

def _quat_from_axis_angle(axis, angle_rad):
    ax = np.array(axis, dtype=np.float32)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    s = np.sin(angle_rad/2.0)
    return np.array([ax[0]*s, ax[1]*s, ax[2]*s, np.cos(angle_rad/2.0)], dtype=np.float32)


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
        """원본 realsense_mpc.py의 Curobo 설정 로직을 따릅니다."""
        self.get_logger().info(f"Loading robot configuration: {self.args.robot}")
        
        self.tensor_args = TensorDeviceType()

        self.world_cfg = WorldConfig.from_dict(
            {
                "blox": {
                    "world": {
                        "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        "integrator_type": "occupancy",
                        "voxel_size": 0.03,
                    }
                }
            }
        )
        
        robot_cfg_dict = load_yaml(join_path(get_robot_configs_path(), self.args.robot))["robot_cfg"]
        
        self.step_dt = 0.02
        self.mpc_config = MpcSolverConfig.load_from_robot_config(
            robot_cfg_dict,
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

        self.cmd_state_full = None
        self.get_logger().info("CuRobo setup is complete.")

    def _setup_ros_communications(self):
        """ROS2 구독자, 발행자, 타이머 설정"""
        # RealSense는 보통 BEST_EFFORT로 발행하므로 맞춰 줌
        img_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
        
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, img_qos)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, img_qos)
        self.create_subscription(CameraInfo, '/camera/camera/depth/camera_info', self.cam_info_callback, img_qos)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.traj_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        # RViz에서 나중에 켜도 최근 맵을 받도록 TransientLocal
        map_qos = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.esdf_pub = self.create_publisher(OccupancyGrid, "/nvblox/esdf_slice", map_qos)

        self.camera_marker_pub = self.create_publisher(Marker, '/camera_pose', 10)

        # 주기 콜백
        self.create_timer(1.0 / 30.0, self.realsense_callback)  # 30Hz: 센서+nvblox 업데이트
        self.create_timer(0.5, self.publish_esdf_slice)         # 2Hz: ESDF 슬라이스 발행
        self.create_timer(1.0 / 50.0, self.mpc_callback)        # 50Hz: MPPI 실행

    # -------------------- (옵션) 테스트용 가짜 깊이 주입 --------------------
    def _inject_fake_depth_square(self, Z=0.60, box_px=80):
        """카메라 정면 Z[m] 위치에 정사각형 장애물 가정한 가짜 depth 생성"""
        if self.depth_image is None or self.camera_intrinsics is None:
            return None
        H, W = self.depth_image.shape
        fake = np.full((H, W), np.inf, dtype=np.float32)
        cy = int(H / 2); cx = int(W / 2); half = box_px // 2
        fake[cy-half:cy+half, cx-half:cx+half] = float(Z)
        return fake
    # -----------------------------------------------------------------------

    def publish_esdf_slice(self):
        """ESDF 수평 슬라이스를 OccupancyGrid로 발행 (RViz 확인용)"""
        try:
            # 슬라이스 파라미터
            res = 0.03
            size_x, size_y = 6.0, 6.0
            width = int(size_x / res)
            height = int(size_y / res)

            # 슬라이스 높이: 카메라 높이(기본값 0.0)
            try:
                tf = self.tf_buffer.lookup_transform('base_link','camera_depth_optical_frame', rclpy.time.Time())
                z0 = float(tf.transform.translation.z)
            except Exception:
                z0 = 0.0

            xs = np.linspace(-size_x/2.0, size_x/2.0, width, dtype=np.float32)
            ys = np.linspace(-size_y/2.0, size_y/2.0, height, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)
            xyz = np.stack([xx, yy, np.full_like(xx, z0)], axis=-1).reshape(-1,3)

            # ---- 실제 ESDF 쿼리 ----
            pts = torch.from_numpy(xyz).to(self.tensor_args.device)
            with torch.no_grad():
                sdf = self.world_model.query_sdf(pts)  # curobo 버전에 맞는 ESDF 쿼리 함수
                
            sdf_np = sdf.detach().cpu().numpy().reshape(height, width)

            # OccupancyGrid 생성
            grid = np.full((height, width), -1, dtype=np.int8)
            grid[sdf_np <= 0.0] = 100  # 장애물/경계
            grid[sdf_np >  0.0] = 0    # 자유

            msg = OccupancyGrid()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"

            info = MapMetaData()
            info.resolution = res
            info.width = width
            info.height = height
            info.origin.position.x = -size_x/2.0
            info.origin.position.y = -size_y/2.0
            info.origin.position.z = 0.0
            info.origin.orientation.w = 1.0
            msg.info = info
            msg.data = grid.flatten().tolist()

            self.esdf_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"publish_esdf_slice failed: {e}", throttle_duration_sec=2.0)

    def depth_callback(self, msg):
        if msg.encoding == "16UC1":
            depth_raw = self.cv_bridge.imgmsg_to_cv2(msg, "16UC1")
            # mm → m
            self.depth_image = depth_raw.astype(np.float32) * 0.001
        else:
            # 32FC1 (이미 m 단위)
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1").astype(np.float32)

    def color_callback(self, msg):
        self.color_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    
    def cam_info_callback(self, msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                "width": msg.width, "height": msg.height,
                "fx": msg.k[0], "fy": msg.k[4], "cx": msg.k[2], "cy": msg.k[5]
            }
            self.get_logger().info("Camera intrinsics received.")
    
    def joint_state_callback(self, msg):
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
        except Exception as e:
            self.get_logger().warn(f"TF lookup for camera failed: {e}", throttle_duration_sec=2.0)
            return

        # ---- (옵션) 가짜 깊이 주입 테스트 ----
        # fake = self._inject_fake_depth_square(Z=0.6, box_px=80)
        # if fake is not None:
        #     self.depth_image = fake

        # ---- 깊이 유효 픽셀 통계 ----
        Z = self.depth_image
        valid = np.isfinite(Z) & (Z > 0.1) & (Z < 5.0)
        if valid.any():
            self.get_logger().info(
                f"Depth valid px: {int(valid.sum())}, zmin={float(Z[valid].min()):.3f}, zmax={float(Z[valid].max()):.3f}"
            )
        else:
            self.get_logger().warn("Depth has no valid pixels in 0.1~5.0m")

        # ---- NVBlox 업데이트 ----
        # self.world_model.decay_layer("world")  # 테스트 중엔 주석 처리 권장
        depth_tensor = self.tensor_args.to_device(torch.from_numpy(self.depth_image))

        fx = self.camera_intrinsics["fx"]; fy = self.camera_intrinsics["fy"]
        cx = self.camera_intrinsics["cx"]; cy = self.camera_intrinsics["cy"]
        intrinsics_tensor = torch.tensor([[fx, 0.0, cx],
                                          [0.0, fy, cy],
                                          [0.0, 0.0, 1.0]],
                                          dtype=torch.float32, device=self.tensor_args.device)

        data_camera = CameraObservation(
            depth_image=depth_tensor.to(dtype=torch.float32),
            intrinsics=intrinsics_tensor,
            pose=camera_pose
        ).to(device=self.tensor_args.device)
        print("world_model methods:", dir(self.world_model))

        self.world_model.add_camera_frame(data_camera, "world")
        self.world_model.process_camera_frames("world", True)  # ESDF 갱신 ON
        torch.cuda.synchronize()

        # ---- ESDF 샘플 로그 ----
        try:
            samples = torch.tensor([[0.0, 0.0, 0.6],
                                    [0.1, 0.0, 0.6],
                                    [-0.1, 0.0, 0.6]],
                                    device=self.tensor_args.device)
            with torch.no_grad():
                sdf_vals = self.world_model.query_sdf(samples)
            self.get_logger().info(f"SDF sample: {sdf_vals.detach().cpu().numpy()}")
        except Exception as e:
            self.get_logger().warn(f"SDF query failed: {e}", throttle_duration_sec=2.0)

        # 창 띄우기 옵션
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
            ordered_positions = []
            for name in self.joint_names:
                idx = self.latest_joint_state_msg.name.index(name)
                ordered_positions.append(self.latest_joint_state_msg.position[idx])
            
            cu_js = CuroboJointState.from_position(
                self.tensor_args.to_device(torch.tensor([ordered_positions]))
            )
            
            if self.cmd_state_full is None:
                self.current_state.copy_(cu_js)
            else:
                self.current_state.copy_(self.cmd_state_full.get_ordered_joint_state(self.joint_names))

        except ValueError:
            self.get_logger().warn("Joint name mismatch in /joint_states.", throttle_duration_sec=5.0)
            return
        
        mpc_result = self.mpc.step(self.current_state, max_attempts=2)

        if 1:
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

    def _publish_camera_marker(self, pose):
        # Pose -> numpy
        p = pose.position.detach().cpu().numpy().reshape(-1)   # [x,y,z]
        q_cam = pose.quaternion.detach().cpu().numpy().reshape(-1)  # [x,y,z,w]

        R_id   = np.array([0,0,0,1], dtype=np.float32)                    # X축
        R_x2y  = _quat_from_axis_angle([0,0,1],  +np.pi/2)                # X→Y
        R_x2z  = _quat_from_axis_angle([0,1,0],  -np.pi/2)                # X→Z

        axes = [
            ((1.0, 0.0, 0.0), R_id,  0),  # X (빨강)
            ((0.0, 1.0, 0.0), R_x2y, 1),  # Y (초록)
            ((0.0, 0.0, 1.0), R_x2z, 2),  # Z (파랑)
        ]

        for (r,g,b), R_extra, idx in axes:
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "camera_pose_axes"
            m.id = idx
            m.type = Marker.ARROW
            m.action = Marker.ADD

            m.pose.position.x = float(p[0])
            m.pose.position.y = float(p[1])
            m.pose.position.z = float(p[2])

            q_draw = _quat_mul(q_cam, R_extra)
            m.pose.orientation.x = float(q_draw[0])
            m.pose.orientation.y = float(q_draw[1])
            m.pose.orientation.z = float(q_draw[2])
            m.pose.orientation.w = float(q_draw[3])

            m.scale.x = 0.1
            m.scale.y = 0.01
            m.scale.z = 0.01

            m.color.a = 1.0
            m.color.r = float(r)
            m.color.g = float(g)
            m.color.b = float(b)

            self.camera_marker_pub.publish(m)


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
