#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import torch
import cv2
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo, JointState, PointCloud2, PointField
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
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.rollout.rollout_base import Goal


def _quat_mul(q1, q2):
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

        # ROS utils
        self.cv_bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Buffers
        self.depth_image = None
        self.color_image = None
        self.camera_intrinsics = None
        self.latest_joint_state_msg = None

        # 성능 파라미터
        self.frame_idx = 0
        self.integrate_every = 3            # 30Hz 카메라 → 10Hz만 적분
        self.nvblox_stride = 2              # NVBlox 입력 다운샘플 배수(2~3 권장)
        self.last_pcl_pub_ns = 0
        self.pcl_pub_period_ns = int(1e9/5) # PCL 5Hz

        # CuRobo & ROS setup
        self._setup_curobo()
        self._setup_ros_communications()
        self.get_logger().info("CuRobo MPC Node initialized successfully.")

    # ---------------- CuRobo setup ----------------
    def _setup_curobo(self):
        self.get_logger().info(f"Loading robot configuration: {self.args.robot}")
        self.tensor_args = TensorDeviceType()

        # base_link 기준 world + occupancy 통합
        self.world_cfg = WorldConfig.from_dict({
            "blox": {
                "world": {
                    "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    "integrator_type": "occupancy",
                    "voxel_size": 0.03,
                }
            }
        })

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

        # initial state/goal
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

    # ---------------- ROS I/O ----------------
    def _setup_ros_communications(self):
        # RealSense 계열은 보통 BEST_EFFORT QoS
        img_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=5)

        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, img_qos)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, img_qos)
        self.create_subscription(CameraInfo, '/camera/camera/depth/camera_info', self.cam_info_callback, img_qos)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.traj_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        # RViz 디버깅용: 카메라 축 표시
        self.camera_marker_pub = self.create_publisher(Marker, '/camera_pose', 10)

        # RViz 디버깅용: depth→PointCloud2 (base_link)
        self.debug_pcl_pub = self.create_publisher(PointCloud2, "/debug/depth_cloud_base", 1)

        # timers
        self.create_timer(1.0/30.0, self.realsense_callback)  # 30 Hz
        self.create_timer(1.0/50.0, self.mpc_callback)        # 50 Hz

    # ---------------- Helpers ----------------
    def _clip_depth_border(self, depth: np.ndarray, h_ratio=0.05, w_ratio=0.05):
        """외곽 노이즈 제거(Isaac clip과 유사)."""
        H, W = depth.shape
        hr = int(H * h_ratio); wr = int(W * w_ratio)
        depth[:hr, :] = 0.0
        depth[-hr:, :] = 0.0
        depth[:, :wr] = 0.0
        depth[:, -wr:] = 0.0

    def _quat_to_rotmat(self, q):
        # q = (x,y,z,w)
        x,y,z,w = q
        xx,yy,zz = x*x, y*y, z*z
        xy,xz,yz = x*y, x*z, y*z
        wx,wy,wz = w*x, w*y, w*z
        return np.array([
            [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
            [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
            [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
        ], dtype=np.float32)

    def _points_to_pointcloud2(self, pts_xyz: np.ndarray, frame_id: str):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = int(pts_xyz.shape[0])
        msg.fields = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = pts_xyz.astype(np.float32).tobytes()
        return msg

    # ---------------- Callbacks ----------------
    def depth_callback(self, msg):
        # 단위 자동 변환
        if msg.encoding == "16UC1":
            depth_raw = self.cv_bridge.imgmsg_to_cv2(msg, "16UC1")
            self.depth_image = depth_raw.astype(np.float32) * 0.001  # mm→m
        else:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1").astype(np.float32)

        # 너무 먼 값 cut (m 단위 기준)
        self.depth_image[self.depth_image > 1.0] = 0.0

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
        start_time = time.time()

        """30Hz: NVBlox 월드 업데이트 + RViz PCL 디버깅"""
        if self.depth_image is None or self.camera_intrinsics is None:
            return

        # 프레임 스키핑(적분 주기 완화)
        self.frame_idx += 1
        integrate_now = (self.frame_idx % self.integrate_every) == 0

        # TF: base_link <- camera_depth_optical_frame
        try:
            trans = self.tf_buffer.lookup_transform('base_link', 'camera_depth_optical_frame', rclpy.time.Time())
            t = trans.transform.translation
            q = trans.transform.rotation
            camera_pose = Pose(
                position=self.tensor_args.to_device(torch.tensor([[t.x, t.y, t.z]], dtype=torch.float32)),
                quaternion=self.tensor_args.to_device(torch.tensor([[q.x, q.y, q.z, q.w]], dtype=torch.float32)),
            )
            self._publish_camera_marker(camera_pose)
        except Exception as e:
            self.get_logger().warn(f"TF lookup for camera failed: {e}", throttle_duration_sec=2.0)
            return

        # 깊이 외곽 클리핑 + 유효 통계
        self._clip_depth_border(self.depth_image)
        Z_full = self.depth_image
        valid = np.isfinite(Z_full) & (Z_full > 0.05) & (Z_full < 5.0)
        if not valid.any():
            return

        # ---------- NVBlox 업데이트 (Isaac과 동일한 occupancy 경로) ----------
        if integrate_now:
            s = self.nvblox_stride
            Z = Z_full[::s, ::s].copy()  # (H/s, W/s)
            K = self.camera_intrinsics
            fx, fy = K["fx"]/s, K["fy"]/s
            cx, cy = K["cx"]/s, K["cy"]/s

            intrinsics_tensor = torch.tensor([[fx, 0.0, cx],
                                              [0.0, fy, cy],
                                              [0.0, 0.0, 1.0]],
                                             dtype=torch.float32, device=self.tensor_args.device)
            depth_tensor = self.tensor_args.to_device(torch.from_numpy(Z).to(dtype=torch.float32))

            data_camera = CameraObservation(
                depth_image=depth_tensor,
                intrinsics=intrinsics_tensor,
                pose=camera_pose
            ).to(device=self.tensor_args.device)

            self.world_model.add_camera_frame(data_camera, "world")
            self.world_model.process_camera_frames("world", False)   # ESDF 안 씀 (Isaac 기본)
            # torch.cuda.synchronize()  # 성능 위해 제거(원하면 주기적으로만)
            self.world_model.update_blox_hashes()                    # ★ MPPI에 최신 occupancy 반영

        # ---------- RViz용: depth → PointCloud2 (base_link) ----------
        try:
            if self.debug_pcl_pub.get_subscription_count() > 0:
                now_ns = self.get_clock().now().nanoseconds
                if now_ns - self.last_pcl_pub_ns >= self.pcl_pub_period_ns:
                    # 시각화는 다운샘플 버전 사용(부하↓)
                    s_vis = max(2, self.nvblox_stride)
                    Z = Z_full[::s_vis, ::s_vis]
                    Hs, Ws = Z.shape
                    yy, xx = np.mgrid[0:Hs, 0:Ws]
                    mask = np.isfinite(Z) & (Z > 0.05) & (Z < 5.0)
                    if mask.any():
                        fx = self.camera_intrinsics["fx"]/s_vis
                        fy = self.camera_intrinsics["fy"]/s_vis
                        cx = self.camera_intrinsics["cx"]/s_vis
                        cy = self.camera_intrinsics["cy"]/s_vis
                        xx = xx[mask].astype(np.float32)
                        yy = yy[mask].astype(np.float32)
                        z  = Z[mask].astype(np.float32)
                        x = (xx - cx) / fx * z
                        y = (yy - cy) / fy * z
                        pts_cam = np.stack([x, y, z], axis=-1)

                        # camera_depth_optical_frame → base_link
                        R = self._quat_to_rotmat((q.x, q.y, q.z, q.w))
                        tvec = np.array([t.x, t.y, t.z], dtype=np.float32)
                        pts_base = (pts_cam @ R.T) + tvec

                        pcl_msg = self._points_to_pointcloud2(pts_base, frame_id="base_link")
                        self.debug_pcl_pub.publish(pcl_msg)
                        self.last_pcl_pub_ns = now_ns
        except Exception as e:
            self.get_logger().warn(f"debug PCL publish failed: {e}", throttle_duration_sec=2.0)

        # (선택) 창으로 원본 보기
        if self.args.show_window and self.color_image is not None:
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((self.color_image, depth_colormap))
            cv2.imshow("RealSense View (ROS)", images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Shutdown requested via CV window.")
                rclpy.shutdown()

        duration = (time.time() - start_time) * 1000.0 # ms
        self.get_logger().info(f"Realsense Callback Duration: {duration:.2f} ms", throttle_duration_sec=1.0)

    def mpc_callback(self):
        start_time = time.time()

        if self.latest_joint_state_msg is None:
            return
        try:
            ordered_positions = []
            for name in self.joint_names:
                idx = self.latest_joint_state_msg.name.index(name)
                ordered_positions.append(self.latest_joint_state_msg.position[idx])

            cu_js = CuroboJointState.from_position(
                self.tensor_args.to_device(torch.tensor([ordered_positions], dtype=torch.float32))
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
            self._fill_and_publish_traj(traj_msg)
        else:
            self.get_logger().warn("MPC step failed.", throttle_duration_sec=1.0)

        duration = (time.time() - start_time) * 1000.0 # ms
        self.get_logger().info(f"MPC Callback Duration: {duration:.2f} ms", throttle_duration_sec=1.0)

    def _fill_and_publish_traj(self, traj_msg):
        traj_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        ordered_cmd = self.cmd_state_full.get_ordered_joint_state(self.joint_names)
        point.positions = ordered_cmd.position.cpu().numpy().flatten().tolist()
        point.velocities = ordered_cmd.velocity.cpu().numpy().flatten().tolist()
        point.time_from_start = Duration(sec=0, nanosec=int(self.step_dt * 1e9))
        traj_msg.points.append(point)
        self.traj_pub.publish(traj_msg)

    def _publish_camera_marker(self, pose):
        p = pose.position.detach().cpu().numpy().reshape(-1)
        q_cam = pose.quaternion.detach().cpu().numpy().reshape(-1)
        R_id  = np.array([0,0,0,1], dtype=np.float32)
        R_x2y = _quat_from_axis_angle([0,0,1],  +np.pi/2)
        R_x2z = _quat_from_axis_angle([0,1,0],  -np.pi/2)
        axes = [((1.0,0.0,0.0), R_id,  0),
                ((0.0,1.0,0.0), R_x2y, 1),
                ((0.0,0.0,1.0), R_x2z, 2)]
        for (r,g,b), R_extra, idx in axes:
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "camera_pose_axes"
            m.id = idx
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose.position.x = float(p[0]); m.pose.position.y = float(p[1]); m.pose.position.z = float(p[2])
            q_draw = _quat_mul(q_cam, R_extra)
            m.pose.orientation.x = float(q_draw[0]); m.pose.orientation.y = float(q_draw[1])
            m.pose.orientation.z = float(q_draw[2]); m.pose.orientation.w = float(q_draw[3])
            m.scale.x = 0.1; m.scale.y = 0.01; m.scale.z = 0.01
            m.color.a = 1.0; m.color.r = float(r); m.color.g = float(g); m.color.b = float(b)
            self.camera_marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="ur5e.yml", help="robot configuration to load")
    parser.add_argument("--show-window", action="store_true", default=False, help="Show CV window")
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
