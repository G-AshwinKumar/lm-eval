#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J myjob            # Job name
#SBATCH -o slurm_output/myjob_%j_mmlu.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/myjob_%j_mmlu.e       # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:1   # Request 1 GPU of 2 available on an average A100 node
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 12:00:00         # Run time (hh:mm:ss) - 30 min
#SBATCH --mem=30G        # Memory per node

MODEL_NAME="mistral7b-full_v1"
PEFT="mistral7b-full_v1_dpo_2"
SAVE_FOLDER="all_med"
echo "Starting sbatch script myjob_arc.sh at `date` for $MODEL_NAME"
MODEL_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/$MODEL_NAME"
SAVE_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/eval_results/$SAVE_FOLDER/$MODEL_NAME.json"
PEFT_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/$PEFT"

module load singularity/3.9.7
if [ -n "$PEFT" ]; then
    SAVE_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/eval_results/$SAVE_FOLDER/${MODEL_NAME}_${PEFT}.json"
    # If PEFT exists, pass it to Singularity
    singularity exec -B /mnt -B /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/lm-evaluation-harness:/home/kike/llm-evaluation-harness --nv /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Singularity/lm-eval-harness_11.8_refactor.sif \
        bash -c 'pip install -U transformers==4.35.2 && export HF_DATASETS_CACHE="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/hf_cache" && \
        python -m lm_eval \
        --model hf \
        --model_args pretrained='"$MODEL_PATH"',peft='"$PEFT_PATH"' \
        --tasks medmcqa_template,medmcqa_val_template,medqa_template,medqa5_template,pubmedqa_template \
        --device cuda:0 \
        --batch_size auto:4 \
        --num_fewshot 3 \
        --output_path "'$SAVE_PATH'"'
else
    SAVE_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/eval_results/$SAVE_FOLDER/${MODEL_NAME}.json"
    singularity exec -B /mnt -B /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/lm-evaluation-harness:/home/kike/llm-evaluation-harness --nv /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Singularity/lm-eval-harness_11.8_refactor.sif \
        bash -c 'pip install -U transformers==4.35.2 && export HF_DATASETS_CACHE="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/hf_cache" && \
        python -m lm_eval \
        --model hf \
        --model_args pretrained='"$MODEL_PATH"' \
        --tasks medmcqa_template,medmcqa_val_template,medqa_template,medqa5_template,pubmedqa_template \
        --device cuda:0 \
        --batch_size auto:4 \
        --num_fewshot 3 \
        --output_path "'$SAVE_PATH'"'
fi