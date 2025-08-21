#!/usr/bin/env python3
"""
Static grasp visualization - exactly like inference but without policy actions
Usage: python visualize_grasp.py <grasp_cache.npy>
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

@hydra.main(config_name="config", config_path="../cfg")
def visualize_grasp(cfg: DictConfig):
    # Get cache file from global variable
    import __main__
    cache_file = __main__.CACHE_FILE
    
    # Load and select random grasp BEFORE setting seed
    # Convert to absolute path if relative
    cache_file_abs = Path(cache_file).resolve()
    grasp_data = np.load(cache_file_abs, allow_pickle=True)
    
    # Use system time for truly random selection
    import time
    np.random.seed(int(time.time() * 1000000) % 2**32)
    grasp_idx = np.random.randint(0, grasp_data.shape[0])
    print(f"Loaded {grasp_data.shape[0]} grasps, selected index {grasp_idx}")
    
    # Configure for single environment visualization
    cfg.test = True
    cfg.num_envs = 1
    cfg.headless = False
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
    
    # Auto-detect object type from filename
    if "cube" in cache_file:
        cfg.task.env.object.type = "cube"
    elif "spray" in cache_file or "bottle" in cache_file:
        cfg.task.env.object.type = "clear_spray_bottle_single"
    
    # Set cache name from filename and extract scale
    cache_name = Path(cache_file).stem
    if "_grasp_50k_s10" in cache_name:
        cfg.task.env.grasp_cache_name = cache_name.replace("_grasp_50k_s10", "")
        cfg.task.env.baseObjScale = 1.0
    elif "_grasp_50k_s095" in cache_name:
        cfg.task.env.grasp_cache_name = cache_name.replace("_grasp_50k_s095", "")
        cfg.task.env.baseObjScale = 0.95
    elif "_grasp_50k_s105" in cache_name:
        cfg.task.env.grasp_cache_name = cache_name.replace("_grasp_50k_s105", "")
        cfg.task.env.baseObjScale = 1.05
    elif "_grasp_50k_s11" in cache_name:
        cfg.task.env.grasp_cache_name = cache_name.replace("_grasp_50k_s11", "")
        cfg.task.env.baseObjScale = 1.1
    elif "_grasp_50k_s09" in cache_name:
        cfg.task.env.grasp_cache_name = cache_name.replace("_grasp_50k_s09", "")
        cfg.task.env.baseObjScale = 0.9
    else:
        cfg.task.env.grasp_cache_name = cache_name
        cfg.task.env.baseObjScale = 1.0
    
    # NOW set the fixed seed for the simulation (after random grasp selection)
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=0)
    
    # Change to leapsim directory so cache files are found
    import os
    os.chdir(Path(__file__).parent.parent)
    
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
    
    print("\nVisualization Controls:")
    print("  Mouse: Rotate view")
    print("  Esc: Quit")
    print("\nShowing static grasp pose")
    
    # Just render loop without stepping policy
    obs = env.reset()
    while True:
        # No actions - just hold the initial grasp pose
        zero_actions = torch.zeros((1, env.num_actions), device=env.device)
        obs, _, _, _ = env.step(zero_actions)
        
        if env.viewer:
            # Check if window closed
            if env.gym.query_viewer_has_closed(env.viewer):
                break

if __name__ == "__main__":
    # Store the cache file and remove it from sys.argv before Hydra processes it
    cache_file_arg = None
    if len(sys.argv) > 1:
        # Find the .npy file in arguments
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg.endswith('.npy'):
                cache_file_arg = arg
                # Remove it from sys.argv
                sys.argv.pop(i)
                break
    
    if not cache_file_arg:
        print("Usage: python visualize_grasp.py <grasp_cache.npy>")
        sys.exit(1)
    
    # Store it globally so the hydra function can access it
    import __main__
    __main__.CACHE_FILE = cache_file_arg
    
    visualize_grasp()