#!/usr/bin/env python3
"""
Static pose visualization task - displays a specific grasp without needing cache files
"""

import numpy as np
import torch
from leapsim.base.vec_task import VecTaskRot
from isaacgym import gymapi, gymtorch

class LeapHandStaticPose(VecTaskRot):
    """Task for visualizing a static hand pose"""
    
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        
        # Override some settings for static visualization
        self.save_init_pose = False
        self.grasp_cache_name = None
        
        super().__init__(cfg=cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        
        # Set up observation and action spaces
        self.num_obs_dict = {
            "full_state": self.num_states
        }
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs_dict["full_state"]), device=self.device, dtype=torch.float)
        
    def _create_envs(self, num_envs, spacing, num_per_row):
        # Create environments with hand and object
        leap_hand_asset = self.gym.load_asset(self.sim, self.asset_root, self.leap_hand_asset_file, self.asset_options)
        leap_hand_start_pose = gymapi.Transform()
        leap_hand_start_pose.p = gymapi.Vec3(0, 0, 0.5)
        leap_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        
        # Load object asset
        object_asset_file = self.asset_files_dict.get(self.object_type, self.asset_files_dict["cube"])
        object_asset = self.gym.load_asset(self.sim, self.asset_root, object_asset_file, self.asset_options)
        
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0174, 0.0179, 0.5533)
        object_start_pose.r = gymapi.Quat(-0.2192, 0.7256, -0.1200, 0.6411)
        
        self.leap_hands = []
        self.envs = []
        self.object_indices = []
        self.hand_indices = []
        
        for i in range(self.num_envs):
            # Create env
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, num_per_row)
            self.envs.append(env)
            
            # Add hand
            leap_hand_actor = self.gym.create_actor(env, leap_hand_asset, leap_hand_start_pose, "leap_hand", i, 1)
            self.leap_hands.append(leap_hand_actor)
            self.hand_indices.append(self.gym.get_actor_index(env, leap_hand_actor, gymapi.DOMAIN_SIM))
            
            # Add object
            object_actor = self.gym.create_actor(env, object_asset, object_start_pose, "object", i, 0)
            self.object_indices.append(self.gym.get_actor_index(env, object_actor, gymapi.DOMAIN_SIM))
            
        self.hand_indices = torch.tensor(self.hand_indices, dtype=torch.int32, device=self.device)
        self.object_indices = torch.tensor(self.object_indices, dtype=torch.int32, device=self.device)
        
    def reset(self, env_ids=None):
        # Set the specific pose
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        # Set hand DOF positions
        self.dof_state[env_ids, 0:4, 0] = torch.tensor([0.1402, -0.9842, 1.6680, -0.1514], device=self.device)  # Index
        self.dof_state[env_ids, 4:8, 0] = torch.tensor([1.1040, 1.5502, 0.9973, 0.8722], device=self.device)   # Middle
        self.dof_state[env_ids, 8:12, 0] = torch.tensor([-0.2385, -0.0831, 1.6133, 0.0843], device=self.device) # Ring
        self.dof_state[env_ids, 12:16, 0] = torch.tensor([-0.0272, 0.8934, 1.6269, 0.2203], device=self.device) # Thumb
        
        # Zero velocities
        self.dof_state[env_ids, :, 1] = 0
        
        # Set object pose
        self.root_state_tensor[self.object_indices[env_ids], 0:3] = torch.tensor([0.0174, 0.0179, 0.5533], device=self.device)
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = torch.tensor([-0.2192, 0.7256, -0.1200, 0.6411], device=self.device)
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = 0  # Zero velocities
        
        # Apply states
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(self.hand_indices), len(env_ids))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(self.object_indices[env_ids]), len(env_ids))
        
        return self.obs_buf
        
    def compute_observations(self):
        self.obs_buf[:] = self.states_buf
        
    def compute_reward(self):
        # No rewards for static visualization
        self.rew_buf[:] = 0
        self.reset_buf[:] = 0