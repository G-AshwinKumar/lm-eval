# lm-evaluation-harness

## Overview
This repo loads EleutherAI/lm-evaluation-harness for llm evaluation, together with the tasks under. Sample scripts are provided in examples/ folder

## Access:

If this is the first time you are using this repo, you need to follow instructions in the following link to get access: https://gitlab.hpai.bsc.es/heka/heka_hub/-/wikis/MLOps-Heka


## Usage - CESGA

### Prepare Script:

Define the model you want to launch:

```
MODEL_NAME="openchat-3.5"
```
This model needs to present at the path `/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/$MODEL_NAME`

Specify the tasks you want to run 

```
    --tasks bold_american_actresses,bold_asian_americans,bold_buddhism...
```

If you want to use both gpus in the node, you need to make sure the following parameters are set correctly:

```
#SBATCH --gres=gpu:a100:2   # Request 2 GPU of 2 available on an average A100 node
#SBATCH -c 64               # 64/64 Cores per task requested
#SBATCH --mem=247G          # Max memory per node

--model_args pretrained='${MODEL_PATH}',tensor_parallel_size=2,...
```
If you want to use only one gpu in the node, your script should look like this:
```
#SBATCH --gres=gpu:a100:1   # Request 1 GPU of 2 available on an average A100 node
#SBATCH -c 32               # Cores per task requested
#SBATCH --mem=120G          # Memory per node

--model_args pretrained='${MODEL_PATH}',tensor_parallel_size=1,...
```
### Git commit

```
git commit -m "LAUNCH_CESGA <sbatch your_script.sh> Whatever message."
```

### Output

Output is saved in CESGA 

```
$SHARE/pipelines/lm-evaluation-harness/$COMMIT_TAG
```

## USAGE - MN5

More on this comming...

## Add a new task

Follow the guide in lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md

Then create a new task inside the /tasks folder of this repo. 

In CESGA, the overlay function for singularity images is disabled, so you will need to build a new image that includes the task you just added. To do this, go to the folder where you cloned this repository and run:
``` 
sudo singularity build <image_name>.sif lm_eval_harness_final.def 
```
Then upload this to cesga and place it in $SHARE/Singularity.

## LLM-Evaluation

We are evaluating 0-shot and 5-shot for a variety of tasks every model in https://gitlab.hpai.bsc.es/heka/heka_hub/-/issues/68.
Pick a model in the list and run either
```
sbatch_vllm_med_and_bias0.sh
sbatch_vllm_med_and_bias5.sh 
```
For models bigger than 7B you'll need more than 6 hours.

We are also evaluating the tasks in huggingface open LLM leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
Pick a model and run
```
sbatch_vllm_hf_leaderboard.sh
```
