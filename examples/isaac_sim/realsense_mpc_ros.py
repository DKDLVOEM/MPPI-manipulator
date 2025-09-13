#!/usr/bin/env python3

import argparse
import copy
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


from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.types.robot import JointState as CuroboJointState
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.curobo_robot_world import CuroboRobotWorld



# CuRobo
from curobo.rollout.rollout_base import Goal
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig


from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader

from helper import VoxelManager, add_robot_to_scene



class CuroboMpcNode(Node):
    def __init__(self, robot_config_name, world_config_path, goal_pose, base_link):
        super().__init__('curobo_mpc_node')

        self.base_link = base_link
        self.goal_pose_list = goal_pose
        self.cv_bridge = CvBridge()
        self.tensor_args = TensorDeviceType()

        self.depth_image = None
        self.intrinsics_data = None
        self.current_joint_state = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.robot_cfg, self.world_cfg, self.mpc_cfg, self.robot_world, self.mpc_solver = \
            self._setup_curobo(robot_config_name, world_config_path)

        self.joint_names = self.robot_cfg.kinematics.joint_names
        
        self._setup_ros_communications()
        
        self.get_logger().info("Curobo MPC Node has been initialized (Final Correction).")

    def _setup_curobo(self, robot_config_name, world_config_path):
        """원본 코드 로직과 100% 동일하게 Curobo 객체들을 초기화합니다."""
        self.get_logger().info("Setting up Curobo with correct initialization...")
        robot_file = join_path(get_robot_configs_path(), robot_config_name)
        world_file = world_config_path
        
        robot_cfg_data = load_yaml(robot_file)["robot_cfg"]
        world_cfg_data = load_yaml(world_file)
        
        robot_cfg = RobotConfig.from_dict(robot_cfg_data)
        world_cfg = WorldConfig.from_dict(world_cfg_data)
        
        #
        # =============================== ERROR FIX ===================================
        # 지적해주신 원본 코드의 MPC 설정 방식을 그대로 사용합니다.
        #
        mpc_cfg = MpcSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            use_cuda_graph=True,
            use_cuda_graph_metrics=True,
            use_cuda_graph_full_step=False,
            self_collision_check=True,
            collision_checker_type=CollisionCheckerType.BLOX,
            use_mppi=True,
            use_lbfgs=False,
            use_es=False,
            store_rollouts=True,
            step_dt=0.02,
        )
        # ===========================================================================
        #
        
        robot_world = CuroboRobotWorld(copy.deepcopy(robot_cfg), world_cfg)
        mpc_solver = MpcSolver(mpc_cfg)

        self.get_logger().info("Curobo setup complete.")
        return robot_cfg, world_cfg, mpc_cfg, robot_world, mpc_solver

    def _setup_ros_communications(self):
        """ROS2 통신 설정."""
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, qos_profile)
        self.cam_info_sub = self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.camera_info_callback, qos_profile)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.joint_trajectory_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        
        self.timer = self.create_timer(1.0 / 60.0, self.physics_step)

    def depth_callback(self, msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        if self.intrinsics_data is None:
            self.intrinsics_data = {
                "width": msg.width, "height": msg.height,
                "fx": msg.k[0], "fy": msg.k[4], "cx": msg.k[2], "cy": msg.k[5],
            }

    def joint_state_callback(self, msg):
        ordered_positions = []
        try:
            for name in self.joint_names:
                idx = msg.name.index(name)
                ordered_positions.append(msg.position[idx])
            self.current_joint_state = CuroboJointState.from_position(
                self.tensor_args.to_device(torch.tensor(ordered_positions, dtype=torch.float32)).view(1, -1)
            )
        except ValueError:
            pass

    def get_camera_to_robot_base_pose(self):
        try:
            transform_stamped = self.tf_buffer.lookup_transform(self.base_link, 'camera_depth_optical_frame', rclpy.time.Time())
            t = transform_stamped.transform.translation
            q = transform_stamped.transform.rotation
            return Pose(
                position=self.tensor_args.to_device(torch.tensor([[t.x, t.y, t.z]])),
                quaternion=self.tensor_args.to_device(torch.tensor([[q.x, q.y, q.z, q.w]]))
            )
        except Exception:
            return None

    def physics_step(self):
        if self.depth_image is None or self.intrinsics_data is None or self.current_joint_state is None:
            return

        cam_pose = self.get_camera_to_robot_base_pose()
        if cam_pose is None: return

        depth_tensor = self.tensor_args.to_device(torch.from_numpy(self.depth_image.astype(np.float32)))
        cam_obs = CameraObservation(depth_image=depth_tensor, intrinsics=self.intrinsics_data)

        self.robot_world.update_world(cam_obs, cam_pose)
        self.mpc_solver.update_world(self.robot_world.get_world_context())

        goal_pose = Pose(
            position=self.tensor_args.to_device(torch.tensor([self.goal_pose_list[0:3]])),
            quaternion=self.tensor_args.to_device(torch.tensor([self.goal_pose_list[3:7]])),
        )
        
        plan = self.mpc_solver.step(self.current_joint_state, goal_pose)

        if plan.is_valid and plan.success.any():
            cmd_state = self.mpc_solver.get_next_step()
            
            traj_msg = JointTrajectory()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.joint_names = self.joint_names
            
            point = JointTrajectoryPoint()
            point.positions = cmd_state.position.cpu().numpy().flatten().tolist()
            point.velocities = cmd_state.velocity.cpu().numpy().flatten().tolist()
            point.time_from_start = Duration(sec=0, nanosec=int(self.mpc_solver.mpc_dt * 1e9))
            
            traj_msg.points.append(point)
            self.joint_trajectory_pub.publish(traj_msg)

            self.current_joint_state = self.mpc_solver.get_current_state()
        else:
            self.get_logger().warn(f"MPC plan is not valid.", throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()


    parser.add_argument("--robot", type=str, default="ur5e.yml", help="robot configuration to load")

    parser.add_argument(
        "--waypoints", action="store_true", help="When True, sets robot in static mode", default=False
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="When True, shows camera image in a CV window",
        default=True,
    )

    parser.add_argument(
        "--use-debug-draw",
        action="store_true",
        help="When True, sets robot in static mode",
        default=False,
    )
    args = parser.parse_args()


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--robot_config", type=str, default="franka.yml", help="robot configuration file to use.")
    # parser.add_argument("--world_config", type=str, default="collision_nvblox.yml", help="World configuration file to use.")
    # parser.add_argument("--goal", nargs=7, type=float, default=[0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], help="Goal pose [x,y,z,qx,qy,qz,qw].")
    # parser.add_argument("--base_link", type=str, default="base_link", help="Robot's base link TF frame name.")

    # cli_args = rclpy.utilities.remove_ros_args(args=sys.argv)[1:]
    # args, _ = parser.parse_known_args(cli_args)
    
    # world_config_path = join_path(get_world_configs_path(), args.world_config)
    
    # node = CuroboMpcNode(
    #     robot_config_name=args.robot_config,
    #     world_config_path=world_config_path,
    #     goal_pose=args.goal,
    #     base_link=args.base_link,
    # )
    

#####################################################################################################################3
    radius = 0.05
    act_distance = 0.4
    voxel_size = 0.05
    render_voxel_size = 0.02
    clipping_distance = 0.7

    # my_world = World(stage_units_in_meters=1.0)
    # stage = my_world.stage

    # stage = my_world.stage
    # my_world.scene.add_default_ground_plane()

    # xform = stage.DefinePrim("/World", "Xform")
    # stage.SetDefaultPrim(xform)
    # target_material = OmniPBR("/World/looks/t", color=np.array([0, 1, 0]))
    # target_material_2 = OmniPBR("/World/looks/t2", color=np.array([0, 1, 0]))
    # if not args.waypoints:
    #     target = cuboid.VisualCuboid(
    #         "/World/target_1",
    #         position=np.array([0.5, 0.0, 0.4]),
    #         orientation=np.array([0, 1.0, 0, 0]),
    #         size=0.04,
    #         visual_material=target_material,
    #     )

    # else:
    #     target = cuboid.VisualCuboid(
    #         "/World/target_1",
    #         position=np.array([0.4, -0.5, 0.2]),
    #         orientation=np.array([0, 1.0, 0, 0]),
    #         size=0.04,
    #         visual_material=target_material,
    #     )

    # # Make a target to follow
    # target_2 = cuboid.VisualCuboid(
    #     "/World/target_2",
    #     position=np.array([0.4, 0.5, 0.2]),
    #     orientation=np.array([0.0, 1, 0.0, 0.0]),
    #     size=0.04,
    #     visual_material=target_material_2,
    # )

    # Make a target to follow
    # camera_marker = cuboid.VisualCuboid(
    #     "/World/camera_nvblox",
    #     position=np.array([-0.05, 0.0, 0.45]),
    #     # orientation=np.array([0.793, 0, 0.609,0.0]),
    #     orientation=np.array([0.5, -0.5, 0.5, -0.5]),
    #     # orientation=np.array([0.561, -0.561, 0.431,-0.431]),
    #     color=np.array([0, 0, 1]),
    #     size=0.01,
    # )

    # NEW: 카메라 위치 명시
    # Make a target to follow
    camera_marker_position = np.array([-0.05, 0.0, 0.45])
    camera_marker_orientation=np.array([0.5, -0.5, 0.5, -0.5])


    # camera_marker.set_visibility(True)
    collision_checker_type = CollisionCheckerType.BLOX
    world_cfg = WorldConfig.from_dict(
        {
            "blox": {
                "world": {
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "integrator_type": "occupancy",
                    "voxel_size": 0.03,
                }
            }
        }
    )
    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] = 0.02
    # robot, _ = add_robot_to_scene(robot_cfg, my_world, "/World/world_robot/")

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_wall.yml"))
    )

    world_cfg_table.cuboid[0].pose[2] -= 0.01
    # usd_help = UsdHelper()

    # usd_help.load_stage(my_world.stage)
    # usd_help.add_world_to_stage(world_cfg_table.get_mesh_world(), base_frame="/World")
    world_cfg.add_obstacle(world_cfg_table.cuboid[0])
    world_cfg.add_obstacle(world_cfg_table.cuboid[1])

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=True,
        use_cuda_graph_metrics=True,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.BLOX,
        use_mppi=True,
        use_lbfgs=False,
        use_es=False,
        store_rollouts=True,
        step_dt=0.02,
    )

    mpc = MpcSolver(mpc_config)

    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    joint_names = mpc.rollout_fn.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg, joint_names=joint_names)
    )
    current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    goal = Goal(
        current_state=current_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=retract_pose,
    )

    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)

    world_model = mpc.world_collision
    realsense_data = RealsenseDataloader(clipping_distance_m=clipping_distance)
    data = realsense_data.get_data()

    camera_pose = Pose.from_list([0, 0, 0, 0.707, 0.707, 0, 0])
    i = 0
    tensor_args = TensorDeviceType()
    # target_list = [target, target_2]
    # target_material_list = [target_material, target_material_2]
    # for material in target_material_list:
    #     material.set_color(np.array([0.1, 0.1, 0.1]))
    target_idx = 0
    cmd_idx = 0
    cmd_plan = None
    # articulation_controller = robot.get_articulation_controller()
    cmd_state_full = None

    cmd_step_idx = 0
    current_error = 0.0
    error_thresh = 0.01
    first_target = False
    if not args.use_debug_draw:
        voxel_viewer = VoxelManager(100, size=render_voxel_size)

    while simulation_app.is_running():
        # my_world.step(render=True)

        # if not my_world.is_playing():
            # if i % 100 == 0:
                # print("**** Click Play to start simulation *****")
            # i += 1
            # if step_index == 0:
            #    my_world.play()
            continue
        # step_index = my_world.current_time_step_index
        # if cmd_step_idx == 0:
            # draw_rollout_points(mpc.get_visual_rollouts(), clear=not args.use_debug_draw)

        # if step_index <= 10:
            # my_world.reset()
            # robot._articulation_view.initialize()
            # idx_list = [robot.get_dof_index(x) for x in j_names]
            # robot.set_joint_positions(default_config, idx_list)

            # robot._articulation_view.set_max_efforts(
                # values=np.array([5000 f/or i in range(len(idx_list))]), joint_indices=idx_list
            # )

        ###########################################################################3
        # TODO:  30hz로 혹은 callback? step_index 대신 timer로 
        # NVBlox 월드 업데이트 (장애물 인식)
        if step_index % 2 == 0.0:
            # camera data updation
            # 카메라 데이터 가져오기
            world_model.decay_layer("world")
            data = realsense_data.get_data()
            # clip_camera(data)
            
            # 카메라의 현재 위치(Pose) 얻기
            # 앞에서 정한 임시 위치
            cube_position, cube_orientation = camera_marker_position, camera_marker_orientation
            camera_pose = Pose(
                position=tensor_args.to_device(cube_position),
                quaternion=tensor_args.to_device(cube_orientation),
            )
            # curobo가 이해하는 형태로 데이터 변환
            data_camera = CameraObservation(  # rgb_image = data["rgba_nvblox"],
                depth_image=data["depth"], intrinsics=data["intrinsics"], pose=camera_pose
            )
            data_camera = data_camera.to(device=tensor_args.device)

            # 월드 모델에 장애물 정보 추가
            world_model.add_camera_frame(data_camera, "world")
            world_model.process_camera_frames("world", False)
            torch.cuda.synchronize()
            world_model.update_blox_hashes()

            bounding = Cuboid("t", dims=[1, 1, 1.0], pose=[0, 0, 0, 1, 0, 0, 0])
            voxels = world_model.get_voxels_in_bounding_box(bounding, voxel_size)
            if voxels.shape[0] > 0:
                voxels = voxels[voxels[:, 2] > voxel_size]
                voxels = voxels[voxels[:, 0] > 0.0]
                # 시각화 화면에 표시하는건데 우린 isaac sim이 아니니까
                # if args.use_debug_draw:
                #     draw_points(voxels)

                # else:
                voxels = voxels.cpu().numpy()
                voxel_viewer.update_voxels(voxels[:, :3])
            else:
                if not args.use_debug_draw:
                    voxel_viewer.clear()
        ###########################################################################


        if args.show_window:
            depth_image = data["raw_depth"]
            color_image = data["raw_rgb"]
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=100), cv2.COLORMAP_VIRIDIS
            )
            color_image = cv2.flip(color_image, 1)
            depth_colormap = cv2.flip(depth_colormap, 1)

            images = np.hstack((color_image, depth_colormap))
            cv2.namedWindow("NVBLOX Example", cv2.WINDOW_NORMAL)
            cv2.imshow("NVBLOX Example", images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
        ######################################################################################
        # TODO: modify from /joint_states
        sim_js = robot.get_joints_state()
        sim_js_names = robot.dof_names
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        ######################################################################################
    
    
        cu_js = cu_js.get_ordered_joint_state(mpc.rollout_fn.joint_names)

        if cmd_state_full is None:
            current_state.copy_(cu_js)
        else:
            current_state_partial = cmd_state_full.get_ordered_joint_state(
                mpc.rollout_fn.joint_names
            )
            current_state.copy_(current_state_partial)
            current_state.joint_names = current_state_partial.joint_names


        # TODO: update goal 필요 시 아래와 같이
        # if current_error <= error_thresh and (not first_target or args.waypoints):
        #     first_target = True
        #     # motion generation:
        #     for ks in range(len(target_material_list)):
        #         if ks == target_idx:
        #             target_material_list[ks].set_color(np.ravel([0, 1.0, 0]))
        #         else:
        #             target_material_list[ks].set_color(np.ravel([0.1, 0.1, 0.1]))

        #     cube_position, cube_orientation = target_list[target_idx].get_world_pose()

        #     # Set EE teleop goals, use cube for simple non-vr init:
        #     ee_translation_goal = cube_position
        #     ee_orientation_teleop_goal = cube_orientation

        #     # compute curobo solution:
        #     ik_goal = Pose(
        #         position=tensor_args.to_device(ee_translation_goal),
        #         quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
        #     )
        #     goal_buffer.goal_pose.copy_(ik_goal)
        #     mpc.update_goal(goal_buffer)
        #     target_idx += 1
        #     if target_idx >= len(target_list):
        #         target_idx = 0

        # TODO 수정하기, 20hz로 만들어서 진행하자
        if cmd_step_idx == 0:
            # mpc step
            mpc_result = mpc.step(current_state, max_attempts=2)
            current_error = mpc_result.metrics.pose_error.item()

        # 다음 스텝의 관절 명령 추출
        cmd_state_full = mpc_result.js_action
        common_js_names = []
        idx_list = []
        for x in sim_js_names:
            if x in cmd_state_full.joint_names:
                # TODO: robot.get_dof_index(x)를 변경 in joint states
                idx_list.append(robot.get_dof_index(x))
                common_js_names.append(x)

        cmd_state = cmd_state_full.get_ordered_joint_state(common_js_names)
        cmd_state_full = cmd_state


        # kju: modify error
        robot._articulation_view.set_joint_position_targets(positions=cmd_state.position.cpu().numpy(), joint_indices=idx_list)
        robot._articulation_view.set_joint_velocity_targets(velocities=cmd_state.velocity.cpu().numpy(), joint_indices=idx_list)
            

        # TODO 수정하기, 20hz로 만들어서 진행하자 1/3
        if cmd_step_idx == 2:
            cmd_step_idx = 0


    realsense_data.stop_device()
    print("finished program")










    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()