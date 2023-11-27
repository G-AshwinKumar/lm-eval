#!/bin/bash

# Set the directory where sbatch scripts are located
sbatch_dir="sbatch_scripts/"

# Function to submit sbatch script for a specific task and model
submit_sbatch() {
    task=$1
    model_name=$2
    sbatch "${sbatch_dir}sbatch_eval_${task}.sh" "$model_name"
}

# Check if task and model_name arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <task> <model_name> [or 'all']"
    exit 1
fi

task=$1
model_name=$2

# Submit sbatch script based on the provided task and model_name
case $task in
    "all")
        for task_name in arc gsm8k mmlu winogrande truthfulqa drop hellaswag; do
            submit_sbatch "$task_name" "$model_name"
            echo "Sbatch script for $task_name with model $model_name submitted."
        done
        ;;
    "arc" | "gsm8k" | "mmlu" | "winogrande" | "truthfulqa" | "drop" | "hellaswag")
        submit_sbatch "$task" "$model_name"
        echo "Sbatch script for $task with model $model_name submitted."
        ;;
    *)
        echo "Invalid task. Supported tasks: arc, gsm8k, mmlu, winogrande, truthfulqa, drop, hellaswag, all."
        exit 1
        ;;
esac
