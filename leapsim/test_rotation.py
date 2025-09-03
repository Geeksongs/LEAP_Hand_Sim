#!/usr/bin/env python3
"""
Test the rotation logic to understand what's happening
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

# Test what happens when we rotate 90 degrees around X axis
print("Testing 90-degree rotation around X axis:")

# Original mug Z-axis (pointing up when mug is upright)
original_z = np.array([0, 0, 1])

# 90 degree rotation around X axis
rot_x_90 = R.from_rotvec([np.pi/2, 0, 0])  # 90 degrees around X
rotated_z = rot_x_90.apply(original_z)

print(f"Original Z-axis: {original_z}")
print(f"After 90° X rotation: {rotated_z}")

# Check angle to vertical
angle_to_vertical = np.arccos(np.clip(np.dot(rotated_z, original_z), -1, 1)) * 180 / np.pi
print(f"Angle to vertical: {angle_to_vertical}°")

print("\nTesting the randomize_rotation_lying_down logic:")
# Simulate what the function does
for i in range(10):
    # Random values like in the function
    rand0 = np.random.uniform(-1, 1)
    rand1 = np.random.uniform(-1, 1) 
    rand2 = np.random.uniform(-1, 1)
    
    # Base 90 degree rotation around X
    base_rot = R.from_rotvec([np.pi/2, 0, 0])
    
    # Small tilts (±36 degrees)
    tilt_x = R.from_rotvec([rand0 * np.pi * 0.2, 0, 0])
    tilt_y = R.from_rotvec([0, rand1 * np.pi * 0.2, 0])
    
    # Z rotation
    rot_z = R.from_rotvec([0, 0, rand2 * np.pi * 2.0])
    
    # Combine: base * tilt_x * tilt_y * rot_z
    final_rot = rot_z * tilt_y * tilt_x * base_rot
    
    # Check final Z-axis direction
    final_z = final_rot.apply(original_z)
    angle_to_vert = np.arccos(np.clip(np.dot(final_z, original_z), -1, 1)) * 180 / np.pi
    
    print(f"Sample {i}: angle to vertical = {angle_to_vert:.1f}°")