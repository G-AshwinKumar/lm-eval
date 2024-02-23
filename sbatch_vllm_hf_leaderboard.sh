#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J eval            # Job name
#SBATCH -o slurm_output/out.txt # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/err.txt # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:1   # Request 1 or 2 GPUs of 2 available on an average A100 node
#SBATCH --exclusive         # No other jobs allowed in our gpu
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 06:00:00         # Run time (hh:mm:ss) - 30 min
#SBATCH --mem=247G          # Memory per node

MODEL_NAME="CausalLM-14B"
echo "Starting sbatch script at `date` for $MODEL_NAME"
MODEL_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/$MODEL_NAME"
# use pwd
CURRENT_DIR=$(pwd)
echo "Current directory: '$CURRENT_DIR'"

module load singularity/3.9.7
singularity exec -B /mnt -B $CURRENT_DIR/tasks:/home/heka_eval/llm-evaluation-harness/lm-eval/tasks --nv /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Singularity/lm_eval_harness_vllm_cuda118_new.sif \
    bash -c 'export HF_DATASETS_CACHE="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/user_caches/hf_cache_'${USER}'" && \
    TORCH_USE_CUDA_DSA=1 python -m lm_eval \
    --model vllm \
    --model_args pretrained="${MODEL_PATH}",tensor_parallel_size=1,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.8 \
    --tasks truthfulqa_mc2 \
    --device cuda \
    --batch_size auto:4 \
    --num_fewshot 0 && \
    TORCH_USE_CUDA_DSA=1 python -m lm_eval \
    --model vllm \
    --model_args pretrained="${MODEL_PATH}",tensor_parallel_size=1,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.8 \
    --tasks mmlu, winogrande, gsm8k \
    --device cuda \
    --batch_size auto:4 \
    --num_fewshot 5 && \
    TORCH_USE_CUDA_DSA=1 python -m lm_eval \
    --model vllm \
    --model_args pretrained="${MODEL_PATH}",tensor_parallel_size=1,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.8 \
    --tasks hellaswag \
    --device cuda \
    --batch_size auto:4 \
    --num_fewshot 10 && \
    TORCH_USE_CUDA_DSA=1 python -m lm_eval \
    --model vllm \
    --model_args pretrained="${MODEL_PATH}",tensor_parallel_size=1,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.8 \
    --tasks arc_challenge \
    --device cuda \
    --batch_size auto:4 \
    --num_fewshot 25'