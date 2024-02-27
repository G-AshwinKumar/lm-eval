#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J eval            # Job name
#SBATCH -o slurm_output/out.txt # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/err.txt # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:1   # Request 1 GPU of 2 available on an average A100 node
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 06:00:00         # Run time (hh:mm:ss)
#SBATCH --mem=247G          # Memory per node

MODEL_NAME="openchat_3.5"
echo "Starting sbatch script at `date` for $MODEL_NAME"
MODEL_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/$MODEL_NAME"
# use pwd
CURRENT_DIR=$(pwd)
echo "Current directory: '$CURRENT_DIR'"

module load singularity/3.9.7
singularity exec -B /mnt -B $CURRENT_DIR/tasks:/home/llm-evaluation-harness/lm_eval/tasks --nv /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Singularity/lm_eval_harness_vllm_cuda118_new.sif \
    bash -c 'export HF_DATASETS_CACHE="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/user_caches/hf_cache_'${USER}'" && \
    CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python -m lm_eval \
    --model hf \
    --model_args pretrained='${MODEL_PATH}',trust_remote_code=True \
    --tasks gsm8k \
    --device cuda \
    --batch_size auto:4 \
    --num_fewshot 0'
