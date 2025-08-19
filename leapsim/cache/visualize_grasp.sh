#!/bin/bash

# Script to visualize grasps from cache files in IsaacGym
# Usage:
#   ./visualize_grasp.sh cube 0        # Visualize grasp 0 from cube cache
#   ./visualize_grasp.sh spray random  # Visualize random grasp from spray bottle cache
#   ./visualize_grasp.sh cube          # Visualize random grasp from cube cache

OBJECT_TYPE=${1:-cube}
GRASP_IDX=${2:-random}

# Set environment variables
export LD_LIBRARY_PATH=/home/python/miniconda3/envs/leapsim/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/js14387/code/LEAP_Hand_Sim/leapsim:$PYTHONPATH

# Determine cache file based on object type
if [[ "$OBJECT_TYPE" == "spray" ]] || [[ "$OBJECT_TYPE" == "spray_bottle" ]]; then
    CACHE_FILE="spray_bottle_grasp_cache_grasp_50k_s10.npy"
    echo "Visualizing spray bottle grasp..."
elif [[ "$OBJECT_TYPE" == "cube" ]]; then
    CACHE_FILE="leap_hand_in_palm_cube_grasp_50k_s10.npy"
    echo "Visualizing cube grasp..."
elif [[ "$OBJECT_TYPE" == "ball" ]]; then
    CACHE_FILE="ball_grasp_cache_grasp_50k_s10.npy"
    echo "Visualizing ball grasp..."
else
    # Assume it's a direct filename
    CACHE_FILE="$OBJECT_TYPE"
    echo "Visualizing grasp from $CACHE_FILE..."
fi

# Build command
CMD="python3 visualize_grasp.py --cache_file $CACHE_FILE"

if [[ "$GRASP_IDX" == "random" ]]; then
    CMD="$CMD --random"
else
    CMD="$CMD --grasp_idx $GRASP_IDX"
fi

echo "Running: $CMD"
echo "================================"

# Run the visualization
cd /home/js14387/code/LEAP_Hand_Sim/leapsim/cache
$CMD