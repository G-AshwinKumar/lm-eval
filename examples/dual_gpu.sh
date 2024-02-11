#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J eval            # Job name
#SBATCH -o slurm_output/out.txt # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/err.txt # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:2   # Request 2 GPU of 2 available on an average A100 node
#SBATCH --exclusive         # No other jobs allowed in our gpu
#SBATCH -c 64               # 64/64 Cores per task requested
#SBATCH -t 06:00:00         # Run time (hh:mm:ss) Set to 6 hours as its the maximum in the short queue. Can be increased to 24 hours in the long queue
#SBATCH --mem=247G          # Max memory per node

MODEL_NAME="meditron-7b"
echo "Starting sbatch script at `date` for $MODEL_NAME"
MODEL_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/$MODEL_NAME"
# use pwd
CURRENT_DIR=$(pwd)
echo "Current directory: '$CURRENT_DIR'"

module load singularity/3.9.7
singularity exec -B /mnt -B $CURRENT_DIR/tasks:/home/heka_eval/llm-evaluation-harness/lm-eval/tasks --nv /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Singularity/lm_eval_harness_vllm_cuda118.sif \
    bash -c 'export HF_DATASETS_CACHE="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/user_caches/hf_cache_'${USER}'" && \
    TORCH_USE_CUDA_DSA=1 python -m lm_eval \
    --model vllm \
    --model_args pretrained='${MODEL_PATH}',tensor_parallel_size=2,trust_remote_code=True,dtype=bfloat16,gpu_memory_utilization=0.9 \
    --tasks toxigen_generation,toxigen_generation_asian,toxigen_generation_black,toxigen_generation_chinese,toxigen_generation_jewish,toxigen_generation_latino,toxigen_generation_lgbtq,toxigen_generation_mental_dis,toxigen_generation_mexican,toxigen_generation_middle_east,toxigen_generation_muslim,toxigen_generation_native_american,toxigen_generation_physical_dis,toxigen_generation_women,bold,bold_american_actors,bold_american_actresses,bold_asian_americans,bold_buddhism,bold_islam,bold_left_wing,bold_hispanic_and_latino_americans,bold_european_americans,bold_african_americans,bold_judaism,bold_atheism,bold_christianity,bold_sikhism,bold_hinduism,truthfulqa_mc2,crows_pairs,hendrycks_ethics,mmlu,medqa_4options,medqa,medqa5,medqa_template,medqa5_template,medmcqa,medmcqa_test,medmcqa_val,medmcqa_template,medmcqa_val_template,pubmedqa \
    --device cuda \
    --batch_size auto:4 \
    --num_fewshot 0'
