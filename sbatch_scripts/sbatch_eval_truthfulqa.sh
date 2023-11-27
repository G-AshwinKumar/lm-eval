#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J myjob            # Job name
#SBATCH -o slurm_output/myjob_%j_truthfulqa.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/myjob_%j_truthfulqa.e       # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:1   # Request 1 GPU of 2 available on an average A100 node
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 06:00:00         # Run time (hh:mm:ss) - 30 min
#SBATCH --mem=30G        # Memory per node

MODEL_NAME="mistral7b-pmc_llama"
TASK="truthfulqa_mc1"

module load singularity/3.9.7
singularity exec -B /mnt -B /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/lm-evaluation-harness:/home/kike/llm-evaluation-harness --nv /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Singularity/lm-eval-harness_11.8_refactor.sif \
    bash -c 'export HF_DATASETS_CACHE="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/hf_cache" && \
    python -m lm_eval \
    --model hf \
    --model_args pretrained=/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/mistral7b-pmc_llama \
    --tasks truthfulqa_mc1 \
    --device cuda:0 \
    --batch_size auto:4 \
    --max_batch_size 64 \
    --num_fewshot 0 \
    --output_path /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/eval_results/truthfulqa_mc1/mistral7b-pmc_llama.json'