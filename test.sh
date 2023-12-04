#!/bin/bash

# Set the directory where sbatch scripts are located
sbatch_dir="sbatch_scripts/"

# Submit each sbatch script by name
sbatch "${sbatch_dir}sbatch_eval_all_med.sh"

echo "All test sbatch scripts submitted."