#!/bin/bash

# Set the directory where sbatch scripts are located
sbatch_dir="sbatch_scripts/"


submit_sbatch() {
    task=$1
    model_name=$2
    peft=$3
    if [ -n "$peft" ]; then
        result_file="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/eval_results/${task}/${model_name}/${peft}.json"
    else
        result_file="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/eval_results/${task}/${model_name}.json"
    fi    
    # Check if the result file already exists
    if [ -f "$result_file" ]; then
        echo "Result file for task '$task' and model '$model_name' and peft "$peft" already exists. Skipping evaluation."
    else
        # Formulate the command based on the presence of the 'peft' argument
        if [ -n "$peft" ]; then
            sbatch "${sbatch_dir}sbatch_eval_${task}.sh" "$model_name" "$peft"
            echo "Sbatch script submitted for task '$task' and model '$model_name' with peft '$peft'."
        else
            sbatch "${sbatch_dir}sbatch_eval_${task}.sh" "$model_name"
            echo "Sbatch script submitted for task '$task' and model '$model_name'."
        fi
    fi
}

# Check if task and model_name and optionally peft arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <task(s)> <model_name>"
    exit 1
fi

tasks=$1
model_name=$2
peft=$3

# Check if "all" is included in the list of tasks
if [[ "$tasks" == *"all"* ]]; then
    for task_name in arc gsm8k mmlu winogrande truthfulqa hellaswag pubmedqa medqa medqa5 medmcqa mmlu_medical ethics; do
        submit_sbatch "$task_name" "$model_name" "$peft"
        echo "Sbatch script for $task_name with model $model_name and peft $peft submitted."
    done
else
    # Submit sbatch script based on the provided task and model_name
    IFS=',' read -ra task_array <<< "$tasks"

    for task in "${task_array[@]}"; do
        case $task in
            "arc" | "gsm8k" | "mmlu" | "winogrande" | "truthfulqa" | "hellaswag" | "pubmedqa" | "medqa" | "medqa5" | "medmcqa" | "mmlu_medical" | "ethics")
                submit_sbatch "$task" "$model_name" "$peft"
                echo "Sbatch script for $task with model $model_name peft $peft submitted."
                ;;
            *)
                echo "Invalid task: $task. Supported tasks: arc, gsm8k, mmlu, winogrande, truthfulqa, hellaswag, pubmedqa, medmcqa, medqa, medqa5, mmlu_medical,ethics, all."
                exit 1
                ;;
        esac
    done
fi
