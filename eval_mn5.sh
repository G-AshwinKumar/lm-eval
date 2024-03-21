#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J eval-mn5      # Job name
#SBATCH -o slurm_output/out.txt       # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/err.txt       # Name of stderr output file(%j expands to jobId)
#SBATCH -n 1
#SBATCH --gres=gpu:1 
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 00:10:00         # Run time (hh:mm:ss) - 30 min
#SBATCH --account bsc70  
#SBATCH --qos=acc_bsccs

MODEL_NAME="mistral7b-full_v2_dpo_merged"
echo "Starting sbatch script at `date` for $MODEL_NAME"
MODEL_PATH="/gpfs/tapes/MN4/projects/bsc70/hpai/storage/data/heka/Models/$MODEL_NAME"
# use pwd
CURRENT_DIR=$(pwd)
echo "Current directory: '$CURRENT_DIR'"

module load singularity
singularity exec --nv /gpfs/scratch/bsc70/hpai/storage/projects/heka/singularity/lm_eval_harness042_vllm032_cuda118_mn5.sif \
    bash -c 'export HF_HUB_OFFLINE=1 && \
    export HF_DATASETS_CACHE="/gpfs/scratch/bsc70/hpai/storage/projects/heka/hf_caches/hf_cache_cns10888" && \
    export HF_HOME="/gpfs/scratch/bsc70/hpai/storage/projects/heka/hf_data" && \
    TORCH_USE_CUDA_DSA=1 python -m lm_eval \
    --model vllm \
    --model_args pretrained='${MODEL_PATH}',tensor_parallel_size=1,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.8 \
    --tasks mir2023_en \
    --device cuda \
    --batch_size auto:4 \
    --num_fewshot 0'
