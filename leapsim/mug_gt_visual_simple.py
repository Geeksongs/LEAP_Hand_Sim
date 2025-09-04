#!/usr/bin/env python3
"""
Simple visualization of target pose for any grasping task
Usage: python mug_gt_visual_simple.py <grasp_file.npy>
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import IsaacGym first, before any torch imports
import isaacgym
from isaacgym import gymapi
from isaacgym import gymtorch

# No torch imports needed for this simple visualization

# Check command line arguments
if len(sys.argv) < 2:
    print("Usage: python mug_gt_visual_simple.py <grasp_file.npy>")
    sys.exit(1)

# Load grasp data from NPY file
grasp_file = sys.argv[1]
grasp_data = np.load(grasp_file, allow_pickle=True)

print(f"Loaded grasp data from {grasp_file}")
print(f"Data shape: {grasp_data.shape}")

# Extract data from NPY file (format: 16 DOFs + 3 pos + 4 quat)
if grasp_data.shape == (23,):
    # Single grasp
    TARGET_HAND_DOF = grasp_data[0:16].tolist()
    TARGET_OBJECT_POS = grasp_data[16:19].tolist()
    TARGET_OBJECT_QUAT = grasp_data[19:23].tolist()
elif len(grasp_data.shape) == 2 and grasp_data.shape[1] == 23:
    # Multiple grasps - use the first one
    TARGET_HAND_DOF = grasp_data[0, 0:16].tolist()
    TARGET_OBJECT_POS = grasp_data[0, 16:19].tolist()
    TARGET_OBJECT_QUAT = grasp_data[0, 19:23].tolist()
    print(f"Multiple grasps found, using grasp 0")
else:
    print(f"Unknown data format: {grasp_data.shape}")
    sys.exit(1)

# Default hand pose (can be overridden by command line in future)
TARGET_HAND_POS = [0.0000, 0.0000, 0.5000]
TARGET_HAND_QUAT = [1.0000, 0.0000, -0.0000, -0.0000]  # [x, y, z, w]

# Auto-detect object type from filename
if "mug" in grasp_file:
    OBJECT_TYPE = "025_mug"
    OBJECT_ASSET = "025_mug.urdf"
elif "airplane" in grasp_file:
    OBJECT_TYPE = "072_toy_airplane" 
    OBJECT_ASSET = "072_toy_airplane.urdf"
elif "meat" in grasp_file:
    OBJECT_TYPE = "010_potted_meat_can"
    OBJECT_ASSET = "010_potted_meat_can.urdf"
elif "tuna" in grasp_file:
    OBJECT_TYPE = "007_tuna_fish_can"
    OBJECT_ASSET = "007_tuna_fish_can.urdf"
elif "scissors" in grasp_file:
    OBJECT_TYPE = "037_scissors"
    OBJECT_ASSET = "037_scissors.urdf"
else:
    # Default to mug
    OBJECT_TYPE = "025_mug"
    OBJECT_ASSET = "025_mug.urdf"

def visualize_target_pose():
    # Initialize gym
    gym = gymapi.acquire_gym()
    
    # Configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 120.0
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)  # No gravity
    
    # Configure PhysX - disable physics
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 0  # Disable physics iterations
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = False  # Use CPU pipeline for better control
    
    # Create sim
    compute_device = 0
    graphics_device = 0
    sim = gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
    
    if sim is None:
        print("Failed to create sim")
        return
        
    # Create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("Failed to create viewer")
        return
    
    # Add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)
    
    # Create environment
    spacing = 1.0
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, lower, upper, 1)
    
    # Load hand asset
    asset_root = "/home/python/Desktop/LEAP_Hand_Sim/assets"
    hand_asset_file = "leap_hand/robot.urdf"
    
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    
    print(f"Loading hand asset from {asset_root}/{hand_asset_file}")
    hand_asset = gym.load_asset(sim, asset_root, hand_asset_file, asset_options)
    
    if hand_asset is None:
        print("Failed to load hand asset!")
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        return
    
    # Load object asset
    object_asset_options = gymapi.AssetOptions()
    
    print(f"Loading {OBJECT_TYPE} asset from {asset_root}/{OBJECT_ASSET}")
    object_asset = gym.load_asset(sim, asset_root, OBJECT_ASSET, object_asset_options)
    
    if object_asset is None:
        print(f"Failed to load {OBJECT_TYPE} asset!")
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        return
    
    # Create hand actor
    hand_pose = gymapi.Transform()
    hand_pose.p = gymapi.Vec3(TARGET_HAND_POS[0], TARGET_HAND_POS[1], TARGET_HAND_POS[2])
    # Normalize hand quaternion
    hand_quat_norm = np.linalg.norm(TARGET_HAND_QUAT)
    normalized_hand_quat = [q/hand_quat_norm for q in TARGET_HAND_QUAT]
    hand_pose.r = gymapi.Quat(normalized_hand_quat[0], normalized_hand_quat[1], normalized_hand_quat[2], normalized_hand_quat[3])
    
    hand_actor = gym.create_actor(env, hand_asset, hand_pose, "hand", 0, 0)
    
    # Create object actor
    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(TARGET_OBJECT_POS[0], TARGET_OBJECT_POS[1], TARGET_OBJECT_POS[2])
    # Normalize quaternion
    quat_norm = np.linalg.norm(TARGET_OBJECT_QUAT)
    normalized_quat = [q/quat_norm for q in TARGET_OBJECT_QUAT]
    object_pose.r = gymapi.Quat(normalized_quat[0], normalized_quat[1], normalized_quat[2], normalized_quat[3])
    
    object_actor = gym.create_actor(env, object_asset, object_pose, OBJECT_TYPE, 1, 0)
    
    # Set hand DOF positions
    num_hand_dofs = gym.get_asset_dof_count(hand_asset)
    hand_dof_props = gym.get_asset_dof_properties(hand_asset)
    
    # Configure DOF properties
    for i in range(num_hand_dofs):
        hand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
        hand_dof_props["stiffness"][i] = 3.0
        hand_dof_props["damping"][i] = 0.1
    
    gym.set_actor_dof_properties(env, hand_actor, hand_dof_props)
    
    # Set target DOF positions
    dof_states = gym.get_actor_dof_states(env, hand_actor, gymapi.STATE_ALL)
    for i in range(min(len(TARGET_HAND_DOF), num_hand_dofs)):
        dof_states['pos'][i] = TARGET_HAND_DOF[i]
        dof_states['vel'][i] = 0.0
    
    gym.set_actor_dof_states(env, hand_actor, dof_states, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, hand_actor, TARGET_HAND_DOF)
    
    # Set camera
    cam_pos = gymapi.Vec3(0.7, 0.7, 0.7)
    cam_target = gymapi.Vec3(0, 0, 0.5)
    gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)
    
    print("\n=== Target Pose Visualization ===")
    print(f"Hand Position: {TARGET_HAND_POS}")
    print(f"Hand Quaternion: {TARGET_HAND_QUAT}")
    print(f"Object Position: {TARGET_OBJECT_POS}")
    print(f"Object Quaternion: {TARGET_OBJECT_QUAT}")
    print(f"Hand DOF Positions:")
    print(f"  Index  finger: {TARGET_HAND_DOF[0:4]}")
    print(f"  Middle finger: {TARGET_HAND_DOF[4:8]}")
    print(f"  Ring   finger: {TARGET_HAND_DOF[8:12]}")
    print(f"  Thumb        : {TARGET_HAND_DOF[12:16]}")
    print("\nPress 'q' or ESC to quit")
    
    # Simulation loop - no physics stepping
    while True:
        # Don't step physics, just render
        # gym.simulate(sim)  # DISABLED - no physics simulation
        # gym.fetch_results(sim, True)  # DISABLED
        
        # Only update graphics
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        
        # Re-apply target positions every frame to override any physics
        gym.set_actor_dof_states(env, hand_actor, dof_states, gymapi.STATE_ALL)
        gym.set_actor_dof_position_targets(env, hand_actor, TARGET_HAND_DOF)
        
        # Check for window closed
        if gym.query_viewer_has_closed(viewer):
            break
            
        # Check for keyboard events
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "quit" and evt.value > 0:
                break
    
    # Cleanup
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    visualize_target_pose()