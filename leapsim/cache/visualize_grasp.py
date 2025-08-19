#!/usr/bin/env python3
"""
Visualize grasp from cache file in IsaacGym environment
Usage:
    python visualize_grasp.py --cache_file spray_bottle_grasp_cache_grasp_50k_s10.npy --grasp_idx 0
    python visualize_grasp.py --cache_file leap_hand_in_palm_cube_grasp_50k_s10.npy --random
"""

import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path to import leapsim modules
sys.path.append(str(Path(__file__).parent.parent))

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
import torch

class GraspVisualizer:
    def __init__(self, cache_file, grasp_idx=None, object_type="cube", headless=False):
        self.cache_file = cache_file
        self.grasp_idx = grasp_idx
        self.object_type = object_type
        self.headless = headless
        
        # Load grasp data
        self.load_grasp_data()
        
        # Initialize gym
        self.gym = gymapi.acquire_gym()
        
        # Parse arguments
        self.args = self.parse_arguments()
        
        # Create sim
        self.create_sim()
        
        # Create environments
        self.create_envs()
        
    def load_grasp_data(self):
        """Load grasp data from npy file"""
        cache_path = Path(__file__).parent / self.cache_file
        if not cache_path.exists():
            print(f"Error: Cache file {cache_path} not found!")
            sys.exit(1)
            
        self.grasp_data = np.load(cache_path, allow_pickle=True)
        print(f"Loaded {self.grasp_data.shape[0]} grasps from {self.cache_file}")
        
        # Select grasp index
        if self.grasp_idx is None:
            self.grasp_idx = np.random.randint(0, self.grasp_data.shape[0])
            print(f"Randomly selected grasp index: {self.grasp_idx}")
        else:
            if self.grasp_idx >= self.grasp_data.shape[0]:
                print(f"Error: Grasp index {self.grasp_idx} out of range (max: {self.grasp_data.shape[0]-1})")
                sys.exit(1)
        
        self.selected_grasp = self.grasp_data[self.grasp_idx]
        print(f"Selected grasp parameters: {self.selected_grasp}")
        
    def parse_arguments(self):
        """Parse command line arguments compatible with IsaacGym"""
        custom_parameters = [
            {"name": "--headless", "action": "store_true", "help": "Run headless"},
            {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments"},
            {"name": "--use_gpu_pipeline", "action": "store_true", "help": "Use GPU pipeline"},
        ]
        args = gymutil.parse_arguments(custom_parameters=custom_parameters)
        args.headless = self.headless
        return args
        
    def create_sim(self):
        """Create simulation"""
        # Configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # Set physics engine
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.5
        sim_params.physx.max_depenetration_velocity = 1000.0
        sim_params.physx.default_buffer_size_multiplier = 5.0
        
        # Create sim
        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params
        )
        
        if self.sim is None:
            print("Failed to create sim")
            sys.exit(1)
            
    def create_envs(self):
        """Create environments with hand and object"""
        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
        # Load assets
        asset_root = str(Path(__file__).parent.parent.parent / "assets")
        
        # Load LEAP hand asset
        hand_asset_file = "leap_hand/robot.urdf"
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.fix_base_link = True
        hand_asset_options.disable_gravity = True
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.collapse_fixed_joints = True
        
        print(f"Loading hand asset from {asset_root}/{hand_asset_file}")
        hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, hand_asset_options)
        
        # Load object asset
        if "cube" in self.object_type:
            object_asset_file = "urdf/cube.urdf"
        elif "spray" in self.object_type or "bottle" in self.cache_file:
            object_asset_file = "clear_spray_bottle_single/clear_spray_bottle_single.urdf"
        else:
            object_asset_file = "urdf/cube.urdf"  # Default to cube
            
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 100.0
        
        print(f"Loading object asset from {asset_root}/{object_asset_file}")
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        
        # Create environment
        spacing = 2.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        env = self.gym.create_env(self.sim, lower, upper, 1)
        
        # Add hand to environment
        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        hand_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        hand_actor = self.gym.create_actor(env, hand_asset, hand_pose, "hand", 0, 1)
        
        # Set hand DOF positions from grasp data
        # Correct format: [joint_angles(16), obj_pos(3), obj_quat(4)]
        if self.selected_grasp.shape[0] >= 23:
            # Get hand DOF properties
            hand_dof_props = self.gym.get_actor_dof_properties(env, hand_actor)
            
            # Set joint positions (parameters 0-15 are joint angles)
            joint_positions = self.selected_grasp[:16]
            
            # Create DOF state tensor
            num_dofs = self.gym.get_actor_dof_count(env, hand_actor)
            dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
            
            for i in range(min(num_dofs, len(joint_positions))):
                dof_states['pos'][i] = joint_positions[i]
                dof_states['vel'][i] = 0.0
                
            self.gym.set_actor_dof_states(env, hand_actor, dof_states, gymapi.STATE_ALL)
            
        # Add object to environment
        object_pose = gymapi.Transform()
        
        # Set object position and orientation from grasp data
        if self.selected_grasp.shape[0] >= 23:
            # Object position (parameters 16-18)
            object_pose.p = gymapi.Vec3(
                self.selected_grasp[16],
                self.selected_grasp[17], 
                self.selected_grasp[18]
            )
            
            # Object orientation (parameters 19-22 as quaternion)
            # Note: Need to normalize if not already normalized
            quat = self.selected_grasp[19:23]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 0:
                quat = quat / quat_norm
            object_pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
        else:
            # Default position if not enough parameters
            object_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            object_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
        object_actor = self.gym.create_actor(env, object_asset, object_pose, "object", 0, 2)
        
        # Set object color
        obj_color = gymapi.Vec3(0.2, 0.4, 0.8)
        self.gym.set_rigid_body_color(env, object_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, obj_color)
        
        # Setup camera
        if not self.headless:
            cam_props = gymapi.CameraProperties()
            cam_props.width = 1280
            cam_props.height = 720
            
            viewer = self.gym.create_viewer(self.sim, cam_props)
            if viewer is None:
                print("Failed to create viewer")
                sys.exit(1)
                
            # Set camera position
            cam_pos = gymapi.Vec3(1.5, 1.5, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
            
            self.viewer = viewer
        else:
            self.viewer = None
            
    def run(self):
        """Run simulation"""
        print("\nVisualization Controls:")
        print("  Space: Pause/Resume simulation")
        print("  R: Reset to initial grasp pose")
        print("  N: Load next grasp")
        print("  P: Load previous grasp")
        print("  Q/Esc: Quit")
        print(f"\nShowing grasp {self.grasp_idx} from {self.cache_file}")
        print(f"Data format: [joint_angles(16), obj_pos(3), obj_quat(4)]")
        print(f"Joint angles: {self.selected_grasp[:16]}")
        print(f"Object pose: pos={self.selected_grasp[16:19]}, quat={self.selected_grasp[19:23]}")
        
        while True:
            # Step simulation
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # Update viewer
            if self.viewer:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                
                # Check for window closed
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
                    
            # Sync frame time
            self.gym.sync_frame_time(self.sim)
            
        # Cleanup
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

def main():
    parser = argparse.ArgumentParser(description="Visualize grasp from cache file")
    parser.add_argument("--cache_file", type=str, required=True,
                       help="Name of the cache file (e.g., spray_bottle_grasp_cache_grasp_50k_s10.npy)")
    parser.add_argument("--grasp_idx", type=int, default=None,
                       help="Index of grasp to visualize (default: random)")
    parser.add_argument("--random", action="store_true",
                       help="Select random grasp")
    parser.add_argument("--object_type", type=str, default="auto",
                       help="Object type (cube, spray_bottle, auto)")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode")
    
    args, unknown = parser.parse_known_args()
    
    # Auto-detect object type from filename
    if args.object_type == "auto":
        if "cube" in args.cache_file:
            args.object_type = "cube"
        elif "spray" in args.cache_file or "bottle" in args.cache_file:
            args.object_type = "spray_bottle"
        else:
            args.object_type = "cube"
    
    # If random flag is set, use None for grasp_idx
    if args.random:
        args.grasp_idx = None
        
    # Create and run visualizer
    visualizer = GraspVisualizer(
        cache_file=args.cache_file,
        grasp_idx=args.grasp_idx,
        object_type=args.object_type,
        headless=args.headless
    )
    
    visualizer.run()

if __name__ == "__main__":
    main()