from isaacgym import gymapi
from isaacgym import gymutil
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Asset and Environment Information")

# get default set of parameters
sim_params = gymapi.SimParams()

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# create sim with these parameters
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

asset_root = "../assets"
# fr5 robot URDF model
asset_file = "urdf/fr5_robot_descripiton/robots/fr5_robot.urdf"

# Example
# asset_file = "urdf/franka_description/robots/franka_panda.urdf"

asset = gym.load_asset(sim, asset_root, asset_file)

spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

env = gym.create_env(sim, lower, upper, 1)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

actor_handle = gym.create_actor(env, asset, pose, "MyActor", 0, 1)

actor_count = gym.get_actor_count(env)
print("%d actors total" % actor_count)

# Iterate through all actors for the environment
for i in range(actor_count):
    actor_handle = gym.get_actor_handle(env, i)
    # print_actor_info(gym, env, actor_handle)

# Cleanup the simulator
gym.destroy_sim(sim)