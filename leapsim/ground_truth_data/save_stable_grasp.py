#!/usr/bin/env python3
"""
Save stable grasp data for screwdriver based on successful hand control
Usage: python save_stable_grasp.py
"""

import numpy as np
from pathlib import Path

# Stable grasp data from Step 40 - when we had good contact
# Hand DOF Positions (16 values)
hand_dof_positions = np.array([
    0.891, -0.815, 1.842, 0.061,  # Index finger
    1.453, 1.417, 0.952, 1.450,  # Middle finger
    0.647, -0.082, 1.744, 0.461,  # Ring finger
    0.514, 0.960, 1.765, 0.640   # Thumb
])

# Object pose (7 values: position + quaternion)
object_pose = np.array([
    -0.0289, 0.0880, 0.5622,     # Object position (x, y, z)
    0.2236, -0.6780, 0.6351, 0.2950  # Object rotation (qx, qy, qz, qw)
])

# Combine hand DOF positions and object pose (23 total values)
# This matches the format expected by grasp cache: [hand_dof_pos (16) + object_root_state (7)]
stable_grasp_data = np.concatenate([hand_dof_positions, object_pose])

print(f"Stable grasp data shape: {stable_grasp_data.shape}")
print(f"Hand DOF positions: {hand_dof_positions}")
print(f"Object pose: {object_pose}")
print(f"Combined data: {stable_grasp_data}")

# Create multiple copies for a small grasp cache (similar to training data format)
num_grasps = 100  # Create 100 copies of this stable grasp
grasp_cache = np.tile(stable_grasp_data, (num_grasps, 1))

print(f"Grasp cache shape: {grasp_cache.shape}")

# Save the grasp cache
output_file = Path(__file__).parent / "screwdriver_stable_grasp_100.npy"
np.save(output_file, grasp_cache)

print(f"Saved stable screwdriver grasp cache to: {output_file}")
print(f"File contains {num_grasps} identical stable grasps")
print(f"Each grasp has {stable_grasp_data.shape[0]} values (16 DOF + 7 pose)")

# Verify the saved file
loaded_data = np.load(output_file)
print(f"Verification - loaded data shape: {loaded_data.shape}")
print(f"First grasp matches: {np.allclose(loaded_data[0], stable_grasp_data)}")