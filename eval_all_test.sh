#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J myjob            # Job name
#SBATCH -o slurm_output/new_eval_all.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/new_eval_all.e       # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:1   # Request 1 GPU of 2 available on an average A100 node
#SBATCH --exclusive         # No other jobs allowed in our gpu
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 00:06:00         # Run time (hh:mm:ss) - 30 min
#SBATCH --mem=247G          # Memory per node

MODEL_NAME="meditron-7b"
echo "Starting sbatch script myjob_arc.sh at `date` for $MODEL_NAME"
MODEL_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/$MODEL_NAME"

module load singularity/3.9.7
singularity exec -B /mnt -B /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/lm-evaluation-harness:/home/kike/llm-evaluation-harness --nv /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Singularity/lm_eval_harness_vllm_cuda118_bias.sif \
    bash -c 'export HF_DATASETS_CACHE="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/hf_cache_'${USER}'" && \
    TORCH_USE_CUDA_DSA=1 python -m lm_eval \
    --model vllm \
    --model_args pretrained='${MODEL_PATH}',tensor_parallel_size=1,trust_remote_code=True,dtype=auto,gpu_memory_utilization=0.9 \
    --tasks list \
    --device cuda:0 \
    --batch_size auto:4 \
    --num_fewshot 0'
