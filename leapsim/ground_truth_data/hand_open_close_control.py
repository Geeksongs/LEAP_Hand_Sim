#!/usr/bin/env python3
"""
Static hand control - based on visualize_grasp.py but with moving fingers
Usage: python hand_open_close_control.py
"""

import sys
import numpy as np
from pathlib import Path
import os

# Use the existing train.py infrastructure
import isaacgym
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from leapsim.utils.reformat import omegaconf_to_dict, print_dict
from leapsim.utils.utils import set_np_formatting, set_seed
import leapsim

@hydra.main(config_name="config", config_path="../cfg", version_base="1.1")
def visualize_hand_control(cfg: DictConfig):
    # Load and select the first grasp from screwdriver cache
    cache_file = "/home/python/Desktop/LEAP_Hand_Sim/leapsim/cache/leap_hand_in_palm_cube_grasp_50k_s10.npy"
    grasp_data = np.load(cache_file, allow_pickle=True)
    
    # Use the first grasp (index 0)
    grasp_idx = 0
    print(f"Loaded {grasp_data.shape[0]} grasps, using index {grasp_idx}")
    
    # Configure for single environment visualization
    cfg.test = True
    cfg.num_envs = 1
    cfg.headless = False  # Visual mode - show the hand with meat can!
    cfg.sim_device = 'cuda:0'
    cfg.rl_device = 'cuda:0'
    cfg.graphics_device_id = 0
    
    # Disable randomization to use only single scale
    cfg.task.env.randomization.randomizeScale = False
    cfg.task.env.randomization.scaleListInit = False
    
    # Set the specific grasp index to use
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    cfg.task.env.sampled_pose_idx = int(grasp_idx)
    OmegaConf.set_struct(cfg, True)
    
    # Set object type to Phillips screwdriver and position it on the palm
    cfg.task.env.object.type = "043_phillips_screwdriver"
    cfg.task.env.grasp_cache_name = "leap_hand_in_palm_cube"  # Use cube cache for now
    cfg.task.env.baseObjScale = 1.0
    
    # Position screwdriver lying down on the palm (above the hand)
    OmegaConf.set_struct(cfg, False)
    cfg.task.env.override_object_init_x = 0.0   # Center on hand
    cfg.task.env.override_object_init_y = 0.1   # Shifted forward from palm center  
    cfg.task.env.override_object_init_z = 0.55  # A bit higher above palm surface (hand is at 0.5)
    # Set initial rotation to lie down (90° around Y axis) + 270° around Z axis (180° more turn)
    # Combined rotation: Y rotation + Z rotation (270° total right turn = 180° from previous)
    cfg.task.env.override_object_init_quat_w = 0.5     # W component for combined rotation
    cfg.task.env.override_object_init_quat_x = 0.5     # X component for combined rotation
    cfg.task.env.override_object_init_quat_y = -0.5    # Negative Y component for 180° flip
    cfg.task.env.override_object_init_quat_z = 0.5     # Z component for combined rotation
    OmegaConf.set_struct(cfg, True)
    
    # NOW set the fixed seed for the simulation
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=0)
    
    # Change to leapsim directory so cache files are found
    # Current file is in ground_truth_data/, parent is leapsim/, parent.parent is LEAP_Hand_Sim/
    current_file = Path(__file__).resolve()  # /home/python/Desktop/LEAP_Hand_Sim/leapsim/ground_truth_data/hand_open_close_control.py
    ground_truth_data_dir = current_file.parent  # /home/python/Desktop/LEAP_Hand_Sim/leapsim/ground_truth_data/
    leapsim_dir = ground_truth_data_dir.parent  # /home/python/Desktop/LEAP_Hand_Sim/leapsim/
    print(f"Current file: {current_file}")
    print(f"Ground truth data dir: {ground_truth_data_dir}")
    print(f"Leapsim dir: {leapsim_dir}")
    
    os.chdir(leapsim_dir)
    print(f"Changed working directory to: {os.getcwd()}")
    
    # Verify cache file exists
    cache_path = Path("cache/leap_hand_in_palm_cube_grasp_50k_s10.npy")
    print(f"Cache file exists: {cache_path.exists()}")
    if not cache_path.exists():
        print(f"Looking for cache files in: {Path('cache').absolute()}")
        if Path('cache').exists():
            cache_files = list(Path('cache').glob('*.npy'))
            print(f"Available cache files: {[f.name for f in cache_files]}")
        else:
            print("Cache directory does not exist!")
    
    
    # Create environment
    env = leapsim.make(
        cfg.seed,
        cfg.task_name,
        cfg.task.env.numEnvs,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
    )
    
    print("\nLEAP Hand with Phillips Screwdriver Control:")
    print("  Mouse: Rotate view")  
    print("  Esc: Quit")
    print("\nShowing smooth hand opening and closing with screwdriver on palm")
    print("Note: Second joint of index, middle and ring fingers are kept steady")
    
    # Reset environment
    obs = env.reset()
    
    # Initialize step counter
    step_counter = 0
    period = 200  # Steps for one complete open-close cycle
    
    # Get the initial grasp pose from the reset
    base_dof_pos = env.leap_hand_dof_pos[0].cpu().numpy().copy()
    
    print(f"Base DOF positions: {base_dof_pos}")
    
    # Define motion ranges relative to the base position
    motion_ranges = np.zeros(16)
    
    # Set motion ranges - much larger for visible opening/closing motion
    motion_ranges[0] = 1.5   # Index base - LARGE motion
    motion_ranges[1] = 0.0   # Index second - STEADY
    motion_ranges[2] = 1.2   # Index third - LARGE motion
    motion_ranges[3] = 1.0   # Index tip - LARGE motion
    
    motion_ranges[4] = 1.5   # Middle base - LARGE motion
    motion_ranges[5] = 0.0   # Middle second - STEADY
    motion_ranges[6] = 1.2   # Middle third - LARGE motion
    motion_ranges[7] = 1.0   # Middle tip - LARGE motion
    
    motion_ranges[8] = 1.5   # Ring base - LARGE motion
    motion_ranges[9] = 0.0   # Ring second - STEADY
    motion_ranges[10] = 1.2  # Ring third - LARGE motion
    motion_ranges[11] = 1.0  # Ring tip - LARGE motion
    
    motion_ranges[12] = 1.5  # Thumb base - LARGE motion
    motion_ranges[13] = 1.0  # Thumb second - can move
    motion_ranges[14] = 1.0  # Thumb third - LARGE motion
    motion_ranges[15] = 0.8  # Thumb tip - LARGE motion
    
    while True:
        # Calculate smooth sinusoidal motion starting from open position
        phase = (step_counter % period) / period * 2 * np.pi
        motion_factor = np.sin(phase + np.pi/2)  # Start from open (add π/2 offset)
        
        # Calculate target DOF positions with much more aggressive motion
        target_dof_pos = base_dof_pos + motion_factor * motion_ranges
        
        # Convert to action instead of directly setting targets
        target_tensor = torch.tensor(target_dof_pos, dtype=torch.float32, device=env.device)
        current_dof = env.leap_hand_dof_pos[0]
        
        # Calculate action as the difference (this should drive the hand to the target)
        # Use much larger action values for visible motion
        action = (target_tensor - current_dof) * 5.0  # Amplify the action
        
        # Clamp actions to reasonable range
        action = torch.clamp(action, -1.0, 1.0)
        
        # Reshape action for environment
        actions = action.unsqueeze(0)  # Add batch dimension
        
        # Step environment with the action
        obs, _, _, _ = env.step(actions)
        
        # Print debug information every 20 steps
        if step_counter % 20 == 0:
            env_id = 0  # Single environment
            hand_dof = env.leap_hand_dof_pos[env_id].cpu().numpy()
            phase_deg = (phase * 180 / np.pi) % 360
            
            # Get hand position from root state tensor
            hand_idx = env.hand_indices[env_id]
            hand_pos = env.root_state_tensor[hand_idx, 0:3].cpu().numpy()
            hand_quat = env.root_state_tensor[hand_idx, 3:7].cpu().numpy()
            
            # Get object position and orientation
            obj_idx = env.object_indices[env_id]
            obj_pos = env.root_state_tensor[obj_idx, 0:3].cpu().numpy()
            obj_quat = env.root_state_tensor[obj_idx, 3:7].cpu().numpy()
            obj_vel = env.root_state_tensor[obj_idx, 7:10].cpu().numpy()  # Linear velocity
            obj_angvel = env.root_state_tensor[obj_idx, 10:13].cpu().numpy()  # Angular velocity
            
            print(f"\n=== Step {step_counter} - Phase: {phase_deg:.1f}° ===")
            print(f"Hand Position: [{hand_pos[0]:.4f}, {hand_pos[1]:.4f}, {hand_pos[2]:.4f}]")
            print(f"Hand Rotation: [{hand_quat[0]:.4f}, {hand_quat[1]:.4f}, {hand_quat[2]:.4f}, {hand_quat[3]:.4f}]")
            print(f"Object Position: [{obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f}]")
            print(f"Object Rotation: [{obj_quat[0]:.4f}, {obj_quat[1]:.4f}, {obj_quat[2]:.4f}, {obj_quat[3]:.4f}]")
            print(f"Object Velocity: [{obj_vel[0]:.4f}, {obj_vel[1]:.4f}, {obj_vel[2]:.4f}]")
            print(f"Object AngVel: [{obj_angvel[0]:.4f}, {obj_angvel[1]:.4f}, {obj_angvel[2]:.4f}]")
            print(f"Motion Factor: {motion_factor:.2f} ({'Opening' if motion_factor > 0 else 'Closing'})")
            
            # Debug: Show targets vs current positions (only first few steps)
            if step_counter < 100 and step_counter % 20 == 0:  # More frequent debug output
                print(f"Target vs Current (Index finger): [{target_dof_pos[0]:.3f} vs {hand_dof[0]:.3f}]")
                print(f"Action magnitude: {torch.norm(action).item():.3f}")
            
            print("Hand DOF Positions:")
            print(f"  Index  finger: [{hand_dof[0]:.3f}, {hand_dof[1]:.3f}*, {hand_dof[2]:.3f}, {hand_dof[3]:.3f}]")
            print(f"  Middle finger: [{hand_dof[4]:.3f}, {hand_dof[5]:.3f}*, {hand_dof[6]:.3f}, {hand_dof[7]:.3f}]")
            print(f"  Ring   finger: [{hand_dof[8]:.3f}, {hand_dof[9]:.3f}*, {hand_dof[10]:.3f}, {hand_dof[11]:.3f}]")
            print(f"  Thumb        : [{hand_dof[12]:.3f}, {hand_dof[13]:.3f}, {hand_dof[14]:.3f}, {hand_dof[15]:.3f}]")
            print("  (* = steady second joint)")
        
        step_counter += 1
        
        if env.viewer:
            # Check if window closed
            if env.gym.query_viewer_has_closed(env.viewer):
                break

if __name__ == "__main__":
    visualize_hand_control()