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


#####################################################################################################################3
    radius = 0.05
    act_distance = 0.4
    voxel_size = 0.05
    render_voxel_size = 0.02
    clipping_distance = 0.7

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


        ###########################################################################3
        # TODO:  30hz로 혹은 callback? step_index 대신 timer로 
        # NVBlox 월드 업데이트 (장애물 인식)
        if step_index % 2 == 0.0:
            # camera data updation
            # 카메라 데이터 가져오기
            world_model.decay_layer("world")
            data = realsense_data.get_data()
            
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