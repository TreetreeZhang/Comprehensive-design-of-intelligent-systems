#!/usr/bin/env python3
# !coding=utf-8

import math
from isaacgym import gymapi
from isaacgym import gymutil

# initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="Load Robot")

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 15
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# create sim with these parameters
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# load robot asset
asset_root = "../assets"
# fr5 robot URDF model
asset_file = "fr5_robot_descripiton/robots/fr5_robot.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Set up the env grid
num_envs = 36
num_per_row = int(math.sqrt(num_envs))
print("Creating %d environments" % num_envs)

spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)

    # add actor
    actor_handle = gym.create_actor(env, robot_asset, pose, "actor", i, 1)

# Create viewer
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# Keep the viewer
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)

# Cleanup the simulator
gym.destroy_sim(sim)
