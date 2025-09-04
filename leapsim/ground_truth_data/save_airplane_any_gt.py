#!/usr/bin/env python3
"""
Save ground truth airplane any grasp data in NPY format
"""

import numpy as np

# Ground truth grasp data for airplane in arbitrary orientation
hand_dof_positions = [
    -0.1971, -0.9989, 1.5916, -0.2213,  # Index finger
    1.1917, 1.6075, 0.8532, 0.9222,      # Middle finger
    -0.0799, 0.0861, 1.5955, 0.0566,     # Ring finger
    -0.1249, 0.7515, 1.5803, 0.1810      # Thumb
]

object_pos = [0.0037, 0.0711, 0.5204]
object_quat = [0.0478, -0.2267, -0.3506, 0.9074]

# Create array in same format as grasp cache files: [16 DOFs, 3 pos, 4 quat] = 23 values
grasp_data = np.array(hand_dof_positions + object_pos + object_quat)

# Save as single grasp
np.save("airplane_any_gt_data.npy", grasp_data)

print("Saved airplane any grasp ground truth to airplane_any_gt_data.npy")
print(f"Data shape: {grasp_data.shape}")
print(f"Data: {grasp_data}")