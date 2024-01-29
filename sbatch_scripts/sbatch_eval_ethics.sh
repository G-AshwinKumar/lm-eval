#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J myjob            # Job name
#SBATCH -o slurm_output/myjob_%j_ethics.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/myjob_%j_ethics.e       # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:1   # Request 1 GPU of 2 available on an average A100 node
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 07:00:00         # Run time (hh:mm:ss) - 30 min
#SBATCH --mem=120G        # Memory per node

# MODEL_NAME="mistral7b-pmc_llama"

MODEL_NAME=$1
PEFT=$2
echo "Starting sbatch script myjob_arc.sh at `date` for $MODEL_NAME and peft $PEFT"
MODEL_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/$MODEL_NAME"
PEFT_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/$PEFT"

module load singularity/3.9.7

# Provide peft argument only if it exists
module load singularity/3.9.7
if [ -n "$PEFT" ]; then
    SAVE_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/eval_results/ethics/${MODEL_NAME}_${PEFT}.json"
    # If PEFT exists, pass it to Singularity
    singularity exec -B /mnt -B /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/lm-evaluation-harness:/home/kike/llm-evaluation-harness --nv /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Singularity/lm-eval-harness_11.8_refactor.sif \
        bash -c 'pip install -U transformers==4.35.2 && export HF_DATASETS_CACHE="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/hf_cache" && \
        python -m lm_eval \
        --model hf \
        --model_args pretrained='"$MODEL_PATH"',peft='"$PEFT_PATH"' \
        --tasks bbh_flan_fewshot,headqa_en,hendrycks_ethics \
        --device cuda:0 \
        --batch_size 2 \
        --num_fewshot 3 \
        --output_path "'$SAVE_PATH'"'
else
    SAVE_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/eval_results/ethics/${MODEL_NAME}.json"
    # If PEFT exists, pass it to Singularity
    singularity exec -B /mnt -B /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/lm-evaluation-harness:/home/kike/llm-evaluation-harness --nv /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Singularity/lm-eval-harness_11.8_refactor.sif \
        bash -c 'pip install -U transformers==4.35.2 && export HF_DATASETS_CACHE="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/hf_cache" && \
        python -m lm_eval \
        --model hf \
        --model_args pretrained='"$MODEL_PATH"' \
        --tasks bbh_flan_fewshot,headqa_en,hendrycks_ethics \
        --device cuda:0 \
        --batch_size 2 \
        --num_fewshot 3 \
        --output_path "'$SAVE_PATH'"'
fi   