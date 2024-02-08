# lm-evaluation-harness

## Overview
This repo loads EleutherAI/lm-evaluation-harness for llm evaluation, together with the tasks under.

## Usage


### Prepare Script:

Define the model you want to launch:

```
MODEL_NAME="openchat-3.5"
```

Specify the tasks you want to run 

```
    --tasks bold_american_actresses,bold_asian_americans,bold_buddhism...
```

If you want multigpu, your script should look like this:

```
#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J myjob            # Job name
#SBATCH -o slurm_output/new_eval_all.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/new_eval_all.e       # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:2   # Request 2 GPU of 2 available on an average A100 node
#SBATCH --exclusive         # No other jobs allowed in our gpu
#SBATCH -c 64               # Cores per task requested

...

    --model_args pretrained='${MODEL_PATH}',tensor_parallel_size=2,...
```

### git commit

```
git commit -m "LAUNCH_CESGA <sbatch your_script.sh> Whatever message."
```


## Output

Output is saved in CESGA 

```
$SHARE/pipelines/lm-evaluation-harness/$COMMIT_TAG
```