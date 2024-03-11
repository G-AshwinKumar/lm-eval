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

MODEL_NAME="Yi-6B-Chat"
echo "Starting sbatch script at `date` for $MODEL_NAME"
MODEL_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/$MODEL_NAME"
# use pwd
CURRENT_DIR=$(pwd)
echo "Current directory: '$CURRENT_DIR'"

module load singularity/3.9.7
singularity exec -B /mnt --nv /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Singularity/lm_eval_harness_vllm032_cuda118.sif \
    bash -c 'export HF_DATASETS_CACHE="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/user_caches/hf_cache_'${USER}'" && \
    TORCH_USE_CUDA_DSA=1 python -m lm_eval \
    --model vllm \
    --model_args pretrained='${MODEL_PATH}',tensor_parallel_size=1,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.7 \
    --tasks multimedqa,toxigen_generation,toxigen_generation_asian,toxigen_generation_black,toxigen_generation_chinese,toxigen_generation_jewish,toxigen_generation_latino,toxigen_generation_lgbtq,toxigen_generation_middle_east,toxigen_generation_mental_dis,toxigen_generation_mexican,toxigen_generation_muslim,toxigen_generation_native_american,toxigen_generation_physical_dis,toxigen_generation_women,bold,bold_american_actors,bold_american_actresses,bold_asian_americans,bold_buddhism,bold_islam,bold_left_wing,bold_hispanic_and_latino_americans,bold_european_americans,bold_african_americans,bold_judaism,bold_atheism,bold_christianity,bold_sikhism,bold_hinduism,truthfulqa_mc2,crows_pairs,hendrycks_ethics,mmlu,medqa,medqa5,medqa_template,medqa5_template,medmcqa,medmcqa_val,medmcqa_val_template \
    --batch_size auto:16 \
    --num_fewshot 0'
