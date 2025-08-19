#!/usr/bin/env python3
"""
Script to check and analyze the quality of generated grasps for spray bottle
Compare with cube grasps to ensure correctness
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_grasp_cache(filepath):
    """Load grasp cache from npy file"""
    data = np.load(filepath, allow_pickle=True)
    return data

def analyze_grasp_distribution(grasps, name=""):
    """Analyze statistical properties of grasp parameters"""
    print(f"\n{'='*60}")
    print(f"Analysis for: {name}")
    print(f"{'='*60}")
    
    print(f"Shape: {grasps.shape}")
    print(f"Number of grasps: {grasps.shape[0]}")
    print(f"Parameters per grasp: {grasps.shape[1]}")
    
    # Statistical analysis
    print(f"\nStatistical Summary:")
    print(f"{'Parameter':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 60)
    
    for i in range(grasps.shape[1]):
        mean = np.mean(grasps[:, i])
        std = np.std(grasps[:, i])
        min_val = np.min(grasps[:, i])
        max_val = np.max(grasps[:, i])
        print(f"Param {i:2d}        {mean:>11.4f}  {std:>11.4f}  {min_val:>11.4f}  {max_val:>11.4f}")
    
    # Check for NaN or Inf values
    nan_count = np.sum(np.isnan(grasps))
    inf_count = np.sum(np.isinf(grasps))
    print(f"\nData Quality Checks:")
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")
    
    return grasps

def compare_grasps(spray_grasps, cube_grasps):
    """Compare spray bottle grasps with cube grasps"""
    print(f"\n{'='*60}")
    print("COMPARISON BETWEEN SPRAY BOTTLE AND CUBE GRASPS")
    print(f"{'='*60}")
    
    # Compare parameter ranges
    print("\nParameter Range Comparison:")
    print(f"{'Param':<8} {'Spray Mean':<12} {'Cube Mean':<12} {'Difference':<12}")
    print("-" * 44)
    
    for i in range(min(spray_grasps.shape[1], cube_grasps.shape[1])):
        spray_mean = np.mean(spray_grasps[:, i])
        cube_mean = np.mean(cube_grasps[:, i])
        diff = spray_mean - cube_mean
        print(f"P{i:2d}      {spray_mean:>11.4f}  {cube_mean:>11.4f}  {diff:>11.4f}")
    
    # Check variance differences
    print("\nVariance Comparison:")
    print(f"{'Param':<8} {'Spray Var':<12} {'Cube Var':<12} {'Ratio':<12}")
    print("-" * 44)
    
    for i in range(min(spray_grasps.shape[1], cube_grasps.shape[1])):
        spray_var = np.var(spray_grasps[:, i])
        cube_var = np.var(cube_grasps[:, i])
        ratio = spray_var / (cube_var + 1e-8)  # Avoid division by zero
        print(f"P{i:2d}      {spray_var:>11.4f}  {cube_var:>11.4f}  {ratio:>11.4f}")

def check_grasp_stability(grasps, name=""):
    """Check if grasps are likely to be stable based on parameter patterns"""
    print(f"\n{'='*60}")
    print(f"STABILITY ANALYSIS for {name}")
    print(f"{'='*60}")
    
    # Check for outliers (values beyond 3 standard deviations)
    outlier_counts = []
    for i in range(grasps.shape[1]):
        mean = np.mean(grasps[:, i])
        std = np.std(grasps[:, i])
        outliers = np.sum(np.abs(grasps[:, i] - mean) > 3 * std)
        outlier_counts.append(outliers)
        if outliers > 0:
            print(f"Parameter {i}: {outliers} outliers (>{3*std:.4f} from mean)")
    
    total_outliers = sum(outlier_counts)
    print(f"\nTotal outliers across all parameters: {total_outliers}")
    
    # Check for duplicate grasps
    unique_grasps = np.unique(grasps, axis=0)
    duplicate_count = grasps.shape[0] - unique_grasps.shape[0]
    print(f"Duplicate grasps: {duplicate_count} ({duplicate_count/grasps.shape[0]*100:.2f}%)")
    
    # Check parameter consistency
    # Assuming first 3 params might be position (x,y,z) and next 4 might be quaternion (qx,qy,qz,qw)
    if grasps.shape[1] >= 7:
        # Check if quaternions are normalized (if params 3-6 are quaternions)
        quat_norms = np.linalg.norm(grasps[:, 3:7], axis=1)
        non_unit_quats = np.sum(np.abs(quat_norms - 1.0) > 0.1)
        print(f"\nQuaternion Analysis (if params 3-6 are quaternions):")
        print(f"  Non-unit quaternions: {non_unit_quats} ({non_unit_quats/grasps.shape[0]*100:.2f}%)")
        print(f"  Mean quaternion norm: {np.mean(quat_norms):.4f}")
        print(f"  Std quaternion norm: {np.std(quat_norms):.4f}")
    
    # Check if joint angles are within reasonable ranges
    # LEAP hand has 16 DOF (4 fingers * 4 joints each)
    if grasps.shape[1] >= 23:  # 7 (pose) + 16 (joints)
        joint_params = grasps[:, 7:]
        print(f"\nJoint Angle Analysis (params 7-22):")
        
        # Check if angles are within typical range (-π to π)
        out_of_range = np.sum((joint_params < -np.pi) | (joint_params > np.pi))
        print(f"  Joint values outside [-π, π]: {out_of_range}")
        
        # Check distribution of joint angles
        print(f"  Mean joint angle: {np.mean(joint_params):.4f} rad")
        print(f"  Std joint angle: {np.std(joint_params):.4f} rad")
        print(f"  Min joint angle: {np.min(joint_params):.4f} rad")
        print(f"  Max joint angle: {np.max(joint_params):.4f} rad")

def visualize_distributions(spray_grasps, cube_grasps):
    """Create visualization of grasp parameter distributions"""
    n_params = min(spray_grasps.shape[1], cube_grasps.shape[1])
    
    # Create subplots for first 8 parameters
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(8, n_params)):
        ax = axes[i]
        
        # Create histograms
        ax.hist(cube_grasps[:, i], bins=30, alpha=0.5, label='Cube', color='blue', density=True)
        ax.hist(spray_grasps[:, i], bins=30, alpha=0.5, label='Spray', color='red', density=True)
        
        ax.set_title(f'Parameter {i}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Grasp Parameter Distributions: Cube vs Spray Bottle')
    plt.tight_layout()
    plt.savefig('/home/js14387/code/LEAP_Hand_Sim/leapsim/cache/grasp_distributions.png', dpi=150)
    print(f"\nVisualization saved to: grasp_distributions.png")

def main():
    # Define file paths
    cache_dir = Path("/home/js14387/code/LEAP_Hand_Sim/leapsim/cache")
    spray_file = cache_dir / "spray_bottle_grasp_cache_grasp_50k_s10.npy"
    cube_file = cache_dir / "leap_hand_in_palm_cube_grasp_50k_s10.npy"
    
    # Load grasp data
    print("Loading grasp cache files...")
    spray_grasps = load_grasp_cache(spray_file)
    cube_grasps = load_grasp_cache(cube_file)
    
    # Analyze individual grasp sets
    analyze_grasp_distribution(spray_grasps, "Spray Bottle Grasps")
    analyze_grasp_distribution(cube_grasps, "Cube Grasps (Reference)")
    
    # Compare grasps
    compare_grasps(spray_grasps, cube_grasps)
    
    # Check stability
    check_grasp_stability(spray_grasps, "Spray Bottle")
    check_grasp_stability(cube_grasps, "Cube (Reference)")
    
    # Visualize distributions
    try:
        visualize_distributions(spray_grasps, cube_grasps)
    except Exception as e:
        print(f"\nCould not create visualization: {e}")
    
    # Final verdict
    print(f"\n{'='*60}")
    print("FINAL ASSESSMENT")
    print(f"{'='*60}")
    
    # Check if spray bottle grasps have similar structure to cube grasps
    similar_shape = spray_grasps.shape == cube_grasps.shape
    similar_ranges = True
    
    for i in range(min(spray_grasps.shape[1], cube_grasps.shape[1])):
        spray_range = np.max(spray_grasps[:, i]) - np.min(spray_grasps[:, i])
        cube_range = np.max(cube_grasps[:, i]) - np.min(cube_grasps[:, i])
        if abs(spray_range - cube_range) / (cube_range + 1e-8) > 2.0:  # More than 200% difference
            similar_ranges = False
            break
    
    print(f"✓ Same data structure: {similar_shape}")
    print(f"✓ Similar parameter ranges: {similar_ranges}")
    
    if similar_shape and similar_ranges:
        print("\n✅ Spray bottle grasps appear to be generated correctly!")
        print("   They follow the same structure and have similar parameter distributions as cube grasps.")
    else:
        print("\n⚠️  Spray bottle grasps may have issues!")
        print("   Further investigation recommended.")

if __name__ == "__main__":
    main()