#!/bin/bash

# Set the MODEL_NAME variable
MODEL_NAME="openchat_3.5"

export MODEL_NAME

# Set the directory where sbatch scripts are located
sbatch_dir="sbatch_scripts/"

# Submit each sbatch script by name
#sbatch "${sbatch_dir}sbatch_eval_all_med.sh"
sbatch "${sbatch_dir}sbatch_vllm_bias0.sh"
# sbatch "${sbatch_dir}sbatch_vllm_med_and_bias0.sh"
# sbatch "${sbatch_dir}sbatch_vllm_med_and_bias5.sh"

echo "All sbatch scripts submitted."
