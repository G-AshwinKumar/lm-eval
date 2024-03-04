#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J eval            # Job name
#SBATCH -o slurm_output/out.txt # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/err.txt # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:1   # Request 1 or 2 GPUs of 2 available on an average A100 node
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 00:10:00         # Run time (hh:mm:ss) - 30 min
#SBATCH --mem=247G          # Memory per node

MODEL_NAME="internlm2-chat-7b"
echo "Starting sbatch script at `date` for $MODEL_NAME"
MODEL_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/$MODEL_NAME"
# use pwd
CURRENT_DIR=$(pwd)
echo "Current directory: '$CURRENT_DIR'"
echo $MODEL_PATH

module load singularity/3.9.7
singularity exec -B /mnt --nv /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Singularity/lm_eval_harness_vllm032_cuda118.sif \
    bash -c 'export HF_DATASETS_CACHE="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/user_caches/hf_cache_'${USER}'" && \
    CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python -m lm_eval \
    --model vllm \
    --model_args pretrained='${MODEL_PATH}',tensor_parallel_size=1,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.85 \
    --tasks winobias,winobias_stereotypical,winobias_anti_stereotypical \
    --device cuda:0 \
    --batch_size auto:4 \
    --num_fewshot 0'
