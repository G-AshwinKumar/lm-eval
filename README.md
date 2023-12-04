## Language Model Evaluation Harness - Big refactor branch (~20/11/2023)

We want to get metrics from [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for general comparison with SOTA models. List of tasks:

- ARC
- HellaSwag
- TruthfulQA
- GSM8K
- Winogrande
- MMLU

For medical tasks, we have MMLU-Medical (from [Meditron](https://arxiv.org/abs/2311.16079)), which is a subset of tasks from MMLU (already in Open LLM Benchmarks)

We add several new medical benchmarks on top: 
- MedQA
- ...

## How to use 

- Each individual benchmark has each own sbatch file in `sbatch_scripts`. They can be run with sbatch `sbatch_scripts/task_name.sh <model_name>`, where model name is the name of the model folder in the path `/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/`. 

- You can run `python model_list.py` to see a list of available models.

- Sbatch files corresponding to OpenLLM Benchmark tasks have the same configuration (num_fewshot) as the leaderboard submissions. 

- Results will be saved to `/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/eval_results/<task_name>` (it can be changed from the sbatch.sh scripts). A new file will be created at `<save_path>/<task_name>/<model_name>.json`  

- To automate the launch of multiple tasks, the script `main_eval.sh <tasks> <model_name>` can be used. It takes the task and the model name as arguments and it creates a new job for each task. If `<task_name>` is set to 'all', a new job will be launched for each task defined in the script. Feel free to create new scripts to automate groups of tasks. Multiple tasks can be passed with format `main_eval.sh task1,task2,task3 <model_name>`

- The `format_results.py <model_name>` will create a .csv file with all results for the selected model in `/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/eval_results/``. It will calculate the average score for Open LLM Leaderboard.


### TODOS
- Save each evaluation for same model-task in separate files to acocunt for variability

