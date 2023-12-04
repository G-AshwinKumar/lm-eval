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
    echo "Usage: $0 <task(s)> <model_name>"
    exit 1
fi

tasks=$1
model_name=$2

# Check if "all" is included in the list of tasks
if [[ "$tasks" == *"all"* ]]; then
    for task_name in arc gsm8k mmlu winogrande truthfulqa hellaswag pubmedqa; do
        submit_sbatch "$task_name" "$model_name"
        echo "Sbatch script for $task_name with model $model_name submitted."
    done
else
    # Submit sbatch script based on the provided task and model_name
    IFS=',' read -ra task_array <<< "$tasks"

    for task in "${task_array[@]}"; do
        case $task in
            "arc" | "gsm8k" | "mmlu" | "winogrande" | "truthfulqa" | "hellaswag" | "pubmedqa")
                submit_sbatch "$task" "$model_name"
                echo "Sbatch script for $task with model $model_name submitted."
                ;;
            *)
                echo "Invalid task: $task. Supported tasks: arc, gsm8k, mmlu, winogrande, truthfulqa, hellaswag, pubmedqa, all."
                exit 1
                ;;
        esac
    done
fi
