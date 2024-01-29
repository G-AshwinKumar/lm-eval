#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J myjob            # Job name
#SBATCH -o slurm_output/myjob_%j_eval_all.o       # Name of stdout output file(%j expands to jobId)
#SBATCH -e slurm_output/myjob_%j_eval_all.e       # Name of stderr output file(%j expands to jobId)
#SBATCH --gres=gpu:a100:1   # Request 2 GPU of 2 available on an average A100 node
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 07:00:00         # Run time (hh:mm:ss) - 30 min
#SBATCH --mem=120G          # Memory per node

if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME is not set."
    exit 1
fi

echo "Starting sbatch script myjob_arc.sh at `date` for $MODEL_NAME"
MODEL_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Models_Trained/llm/$MODEL_NAME"
SAVE_PATH="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/eval_results/eval_all/$MODEL_NAME.json"

module load singularity/3.9.7
singularity exec -B /mnt -B /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/lm-evaluation-harness:/home/kike/llm-evaluation-harness --nv /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Singularity/lm_eval_harness_vllm_cuda118.sif \
    bash -c 'export HF_DATASETS_CACHE="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/hf_cache_'${USER}'" && \
    TORCH_USE_CUDA_DSA=1 python -m lm_eval \
    --model vllm \
    --model_args pretrained='${MODEL_PATH}',tensor_parallel_size=1,trust_remote_code=True,dtype=auto,gpu_memory_utilization=0.6 \
    --tasks mmlu,medqa_4options,medqa,medqa5,medqa_template,medqa5_template,medmcqa,medmcqa_test,medmcqa_val,medmcqa_template,medmcqa_val_template,pubmedqa \
    --device cuda:0 \
    --batch_size auto:4 \
    --num_fewshot 5 \
    --output_path "'$SAVE_PATH'"'

