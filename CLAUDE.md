# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the LEAP Hand simulation environment for low-cost, efficient, and anthropomorphic robotic hand learning. It uses IsaacGym for physics simulation and implements reinforcement learning for in-hand object manipulation tasks.

## Key Commands

### Environment Setup
```bash
# Activate conda environment
conda activate leapsim

# Set required environment variables for training
export LD_LIBRARY_PATH=/home/python/miniconda3/envs/leapsim/lib:$LD_LIBRARY_PATH
```

### Training Commands

**Train cube manipulation (single GPU):**
```bash
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/home/python/miniconda3/envs/leapsim/lib:$LD_LIBRARY_PATH python3 leapsim/train.py task=LeapHandRot max_iterations=1000 task.env.grasp_cache_name=leap_hand_in_palm_cube task.env.object.type=cube num_envs=12288 wandb_activate=false
```

**Train spray bottle manipulation:**
```bash
CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH=/home/python/miniconda3/envs/leapsim/lib:$LD_LIBRARY_PATH python3 leapsim/train.py task=LeapHandRot max_iterations=1000 task.env.grasp_cache_name=spray_bottle_grasp_cache task.env.object.type=clear_spray_bottle_single task.env.baseObjScale=1.0 num_envs=12288 wandb_activate=false
```

**Generate grasp cache for training:**
```bash
# For cube at different scales
for cube_scale in 0.9 0.95 1.0 1.05 1.1; do
    bash leapsim/scripts/gen_grasp.sh $cube_scale custom_grasp_cache num_envs=1024 wandb_activate=false
done

# For spray bottle
bash leapsim/scripts/gen_grasp.sh 1.0 spray_bottle_grasp_cache num_envs=1024 wandb_activate=false
```

**Test/visualize trained policy:**
```bash
python3 leapsim/train.py wandb_activate=false num_envs=1 headless=false test=true task=LeapHandRot checkpoint=runs/<checkpoint_name>/nn/LeapHand.pth
```

**Deploy to real robot:**
```bash
python3 leapsim/deploy.py wandb_activate=false num_envs=1 headless=false test=true task=LeapHandRot checkpoint=runs/<checkpoint_name>/nn/LeapHand.pth
```

### Important Configuration Parameters

- `num_envs`: Number of parallel environments (default: 12288 for training)
- `max_iterations`: Training iterations (default: 1000-5000)
- `task.env.grasp_cache_name`: Name of grasp cache file (must match generated cache)
- `task.env.object.type`: Object type ('cube', 'clear_spray_bottle_single', etc.)
- `task.env.baseObjScale`: Object scale factor (default: 1.0)
- `wandb_activate`: Enable Weights & Biases logging (default: false for local runs)
- `headless`: Run without visualization (true for training, false for testing)
- `test`: Run in test/inference mode (requires checkpoint)

## Architecture Overview

### Task Hierarchy
```
VecTask (base/vec_task.py)
  └── VecTaskRot (base/vec_task.py) - Adds rotation-specific functionality
      └── LeapHandRot (tasks/leap_hand_rot.py) - Main manipulation task
          └── LeapHandGrasp (tasks/leap_hand_grasp.py) - Grasp generation task
```

### Key Components

**Training Pipeline:**
- `leapsim/train.py`: Main training entry point using Hydra configuration
- `leapsim/learning/common_agent.py`: RL agent implementation  
- `leapsim/learning/amp_continuous.py`: PPO algorithm implementation
- Learning rate: Set in `leapsim/cfg/train/LeapHandRotPPO.yaml` (currently 1e-3)

**Environment:**
- `leapsim/tasks/leap_hand_rot.py`: Core task implementation with reward functions
- Grasp cache files: Located in `leapsim/cache/` directory
- Object assets: Located in `assets/` directory

**Configuration:**
- Main config: `leapsim/cfg/config.yaml`
- Task config: `leapsim/cfg/task/LeapHandRot.yaml`
- Training config: `leapsim/cfg/train/LeapHandRotPPO.yaml`

### Reward Structure
The task uses multiple reward components:
- `rotate_finite_diff`: Reward for object rotation (weight: 1.25)
- `object_fallen`: Penalty when object drops (weight: -10)
- `objLinvelPenaltyScale`: Penalty for object linear velocity (-0.3)
- `poseDiffPenaltyScale`: Penalty for deviation from canonical pose (-0.1)
- `torquePenaltyScale`: Penalty for high torques (-0.1)

### Multi-GPU Training Note
Multi-GPU training with `torchrun` requires removing horovod dependency. The code has been modified to use `LOCAL_RANK` environment variable instead of horovod for multi-GPU support, but IsaacGym has limitations with multi-GPU that may cause CUDA alignment errors.

### Common Issues

1. **CUDA misaligned address error**: Use single GPU instead of multi-GPU training
2. **Module not found errors**: Ensure all dependencies are installed and conda environment is activated
3. **Grasp cache not found**: Generate grasp cache before training using gen_grasp.sh script
4. **Low FPS during training**: Reduce num_envs or use more powerful GPU

## Working Directory
Always run commands from `/home/js14387/code/LEAP_Hand_Sim/leapsim/` directory unless specified otherwise.