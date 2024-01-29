#!/bin/bash

# Set the MODEL_NAME variable
MODEL_NAME="Mixtral-8x7B-v0.1"

export MODEL_NAME

# Set the directory where sbatch scripts are located
sbatch_dir="sbatch_scripts/"

# Submit each sbatch script by name
#sbatch "${sbatch_dir}sbatch_eval_all_med.sh"
sbatch "${sbatch_dir}sbatch_vllm.sh"

echo "All sbatch scripts submitted."
