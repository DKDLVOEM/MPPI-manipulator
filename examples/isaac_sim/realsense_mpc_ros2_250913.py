#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import torch
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

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

        # CuRobo & ROS setup
        self._setup_curobo()
        self._setup_ros_communications()
        self.get_logger().info("CuRobo MPC Node initialized successfully.")

    # ---------------- CuRobo setup ----------------
    def _setup_curobo(self):
        self.get_logger().info(f"Loading robot configuration: {self.args.robot}")
        self.tensor_args = TensorDeviceType()

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
        img_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=5)

        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, img_qos)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, img_qos)
        self.create_subscription(CameraInfo, '/camera/camera/depth/camera_info', self.cam_info_callback, img_qos)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.traj_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        # PointCloud2 publishers
        self.pcl_pub = self.create_publisher(PointCloud2, "/realsense/points", 1)  # RViz 확인용
        self.octo_pub = self.create_publisher(PointCloud2, "/cloud_in", 1)         # octomap_server 입력 기본값

        # Markers (camera axes)
        self.camera_marker_pub = self.create_publisher(Marker, '/camera_pose', 10)

        # timers
        self.create_timer(1.0/30.0, self.realsense_callback)  # 30 Hz (sensor + curobo update)
        self.create_timer(1.0/50.0, self.mpc_callback)        # 50 Hz

    # ---------------- Utils ----------------
    def _depth_to_points(self, depth_m: np.ndarray, fx, fy, cx, cy, stride: int = 2):
        """depth[m] -> (N,3) XYZ in camera_depth_optical_frame. stride로 다운샘플."""
        H, W = depth_m.shape
        valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 5.0)
        yy, xx = np.mgrid[0:H:stride, 0:W:stride]
        z = depth_m[0:H:stride, 0:W:stride]
        mask = valid[0:H:stride, 0:W:stride]
        if not mask.any():
            return np.empty((0,3), dtype=np.float32)
        xx = xx[mask]; yy = yy[mask]; z = z[mask]
        x = (xx - cx) / fx * z
        y = (yy - cy) / fy * z
        pts = np.stack([x, y, z], axis=-1).astype(np.float32)
        return pts

    def _points_to_pointcloud2(self, pts_xyz: np.ndarray, frame_id: str):
        """(N,3) float32 -> sensor_msgs/PointCloud2"""
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
        msg.data = pts_xyz.tobytes()
        return msg

    # ---------------- Callbacks ----------------
    def depth_callback(self, msg):
        # 안전하게 단위 자동 변환
        if msg.encoding == "16UC1":
            depth_raw = self.cv_bridge.imgmsg_to_cv2(msg, "16UC1")
            self.depth_image = depth_raw.astype(np.float32) * 0.001  # mm→m
        else:
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
        """30Hz: NVBlox 업데이트 + OctoMap 입력 PCL 퍼블리시"""
        if self.depth_image is None or self.camera_intrinsics is None:
            return

        # TF: camera pose (원하면 생략 가능)
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
            camera_pose = None  # NVBlox 업데이트만 실패시 생략해도 됨

        # 깊이 통계
        Z = self.depth_image
        valid = np.isfinite(Z) & (Z > 0.05) & (Z < 5.0)
        if valid.any():
            self.get_logger().info(f"Depth valid px: {int(valid.sum())}, zmin={float(Z[valid].min()):.3f}, zmax={float(Z[valid].max()):.3f}")
        else:
            self.get_logger().warn("Depth has no valid pixels in 0.05~5.0m")

        # ---- Depth -> PointCloud2 (RViz & OctoMap) ----
        fx = self.camera_intrinsics["fx"]; fy = self.camera_intrinsics["fy"]
        cx = self.camera_intrinsics["cx"]; cy = self.camera_intrinsics["cy"]
        pts = self._depth_to_points(self.depth_image, fx, fy, cx, cy, stride=2)
        if pts.shape[0] > 0:
            pcl_msg = self._points_to_pointcloud2(pts, frame_id="camera_depth_optical_frame")
            # RViz 확인용
            self.pcl_pub.publish(pcl_msg)
            # OctoMap 입력 (octomap_server의 기본 구독 토픽)
            self.octo_pub.publish(pcl_msg)

        # ---- (선택) NVBlox 업데이트 유지하고 싶으면 아래 사용 ----
        if camera_pose is not None:
            try:
                depth_tensor = self.tensor_args.to_device(torch.from_numpy(self.depth_image))
                K = self.camera_intrinsics
                intrinsics_tensor = torch.tensor([[K["fx"], 0.0,      K["cx"]],
                                                  [0.0,      K["fy"], K["cy"]],
                                                  [0.0,      0.0,      1.0]],
                                                  dtype=torch.float32, device=self.tensor_args.device)
                data_camera = CameraObservation(
                    depth_image=depth_tensor.to(dtype=torch.float32),
                    intrinsics=intrinsics_tensor,
                    pose=camera_pose
                ).to(device=self.tensor_args.device)
                self.world_model.add_camera_frame(data_camera, "world")
                self.world_model.process_camera_frames("world", True)
                torch.cuda.synchronize()
            except Exception as e:
                self.get_logger().warn(f"NVBlox update failed: {e}", throttle_duration_sec=2.0)

        # (옵션) 창 표시
        if self.args.show_window and self.color_image is not None:
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((self.color_image, depth_colormap))
            cv2.imshow("RealSense View (ROS)", images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Shutdown requested via CV window.")
                rclpy.shutdown()

    def mpc_callback(self):
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
    parser.add_argument("--robot", type=str, default="ur5e.yml")
    parser.add_argument("--show-window", action="store_true", default=False)
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
