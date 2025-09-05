#!/usr/bin/env python3
"""
Save ground truth screwdriver stable grasp data in NPY format
"""

import numpy as np

# Ground truth stable grasp data for screwdriver from Step 40
hand_dof_positions = [
    0.891, -0.815, 1.842, 0.061,  # Index finger
    1.453, 1.417, 0.952, 1.450,   # Middle finger
    0.647, -0.082, 1.744, 0.461,  # Ring finger
    0.514, 0.960, 1.765, 0.640    # Thumb
]

object_pos = [-0.0289, 0.0880, 0.5622]
object_quat = [0.2236, -0.6780, 0.6351, 0.2950]

# Create array in same format as grasp cache files: [16 DOFs, 3 pos, 4 quat] = 23 values
grasp_data = np.array(hand_dof_positions + object_pos + object_quat)

# Save as single grasp
np.save("screwdriver_stable_gt_data.npy", grasp_data)

print("Saved screwdriver stable grasp ground truth to screwdriver_stable_gt_data.npy")
print(f"Data shape: {grasp_data.shape}")
print(f"Data: {grasp_data}")
print(f"\nHand DOF positions: {hand_dof_positions}")
print(f"Object position: {object_pos}")
print(f"Object quaternion: {object_quat}")