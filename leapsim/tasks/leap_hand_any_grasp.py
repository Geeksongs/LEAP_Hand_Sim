# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Task for training RL model to reach ground truth pose from random initial poses
# --------------------------------------------------------

import torch
import numpy as np
from .leap_hand_rot import LeapHandRot

class LeapHandAnyGrasp(LeapHandRot):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=None, force_render=None):
        # Call parent class initialization
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        
        # Load target pose from ground truth NPY file (after parent init so self.device is available)
        if 'target_gt_file' in self.cfg['env']:
            gt_data = np.load(self.cfg['env']['target_gt_file'])
            self.target_hand_dof = torch.tensor(gt_data[0:16], device=self.device, dtype=torch.float)
            self.target_object_pos = torch.tensor(gt_data[16:19], device=self.device, dtype=torch.float)
            self.target_object_quat = torch.tensor(gt_data[19:23], device=self.device, dtype=torch.float)
            print(f"Loaded target pose from {self.cfg['env']['target_gt_file']}")
        else:
            # Fallback to hardcoded values if no GT file specified
            self.target_hand_dof = torch.tensor([
                0.1402, -0.9842, 1.6680, -0.1514,  # Index
                1.1040, 1.5502, 0.9973, 0.8722,   # Middle
                -0.2385, -0.0831, 1.6133, 0.0843,  # Ring
                -0.0272, 0.8934, 1.6269, 0.2203    # Thumb
            ], device=self.device, dtype=torch.float)
            self.target_object_pos = torch.tensor([0.0174, 0.0179, 0.5533], device=self.device, dtype=torch.float)
            self.target_object_quat = torch.tensor([-0.2192, 0.7256, -0.1200, 0.6411], device=self.device, dtype=torch.float)

    def reward_object_position_reward(self):
        """Dense reward for object position - closer to target gets higher reward"""
        target_pos = self.target_object_pos.unsqueeze(0).expand(self.num_envs, -1)
        pos_diff = torch.norm(self.object_pos - target_pos, dim=-1)
        # Exponential reward - closer positions get much higher rewards
        reward = torch.exp(-5.0 * pos_diff)  # Scale factor can be tuned
        return reward
    
    def reward_object_orientation_reward(self):
        """Dense reward for object orientation - closer to target orientation gets higher reward"""
        target_quat = self.target_object_quat.unsqueeze(0).expand(self.num_envs, -1)
        # Normalize target quaternion
        target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)
        
        # Calculate quaternion difference
        # Using dot product to measure similarity (1.0 = identical, 0.0 = orthogonal)
        quat_similarity = torch.abs(torch.sum(self.object_rot * target_quat, dim=-1))
        # Convert to angular difference (0 = identical, pi = opposite)
        angular_diff = 2.0 * torch.acos(torch.clamp(quat_similarity, max=1.0))
        # Exponential reward - closer orientations get higher rewards  
        reward = torch.exp(-2.0 * angular_diff)  # Scale factor can be tuned
        return reward
    
    def reward_object_pose_reward(self):
        """Dense reward for combined object pose (position + orientation)"""
        # Position component
        target_pos = self.target_object_pos.unsqueeze(0).expand(self.num_envs, -1)
        pos_diff = torch.norm(self.object_pos - target_pos, dim=-1)
        pos_reward = torch.exp(-5.0 * pos_diff)
        
        # Orientation component  
        target_quat = self.target_object_quat.unsqueeze(0).expand(self.num_envs, -1)
        target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)
        quat_similarity = torch.abs(torch.sum(self.object_rot * target_quat, dim=-1))
        angular_diff = 2.0 * torch.acos(torch.clamp(quat_similarity, max=1.0))
        orient_reward = torch.exp(-2.0 * angular_diff)
        
        # Combined reward - both position and orientation must be good
        # Using geometric mean to ensure both components contribute
        combined_reward = torch.sqrt(pos_reward * orient_reward)
        
        return combined_reward
    
    def reward_hand_dof_reward(self):
        """Dense reward for hand DOF positions - closer to target gets higher reward"""
        target_dof = self.target_hand_dof.unsqueeze(0).expand(self.num_envs, -1)
        dof_diff = torch.norm(self.leap_hand_dof_pos - target_dof, dim=-1)
        # Exponential reward - closer positions get higher rewards
        reward = torch.exp(-1.0 * dof_diff)  # Scale factor can be tuned
        return reward
    
    def reward_success_bonus(self):
        """Bonus reward when both object and hand reach target poses"""
        # Check if object position is close enough (within 2cm)
        target_pos = self.target_object_pos.unsqueeze(0).expand(self.num_envs, -1)
        pos_close = torch.norm(self.object_pos - target_pos, dim=-1) < 0.02
        
        # Check if object orientation is close enough (within 10 degrees)
        target_quat = self.target_object_quat.unsqueeze(0).expand(self.num_envs, -1)
        target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)
        quat_similarity = torch.abs(torch.sum(self.object_rot * target_quat, dim=-1))
        angular_diff = 2.0 * torch.acos(torch.clamp(quat_similarity, max=1.0))
        orient_close = angular_diff < 0.174  # 10 degrees in radians
        
        # Check if hand DOF is close enough 
        target_dof = self.target_hand_dof.unsqueeze(0).expand(self.num_envs, -1)
        dof_close = torch.norm(self.leap_hand_dof_pos - target_dof, dim=-1) < 0.1
        
        # All conditions must be met for success bonus
        success = pos_close & orient_close & dof_close
        return success.float()