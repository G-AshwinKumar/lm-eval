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
#SBATCH -t 05:00:00         # Run time (hh:mm:ss) - 30 min
#SBATCH --account bsc70  
#SBATCH --qos=acc_bsccs

MODEL_NAME="mistral7b-full_v6_med"
echo "Starting sbatch script at `date` for $MODEL_NAME"
MODEL_PATH="/gpfs/tapes/MN4/projects/bsc70/hpai/storage/data/heka/Models/$MODEL_NAME"
# use pwd
CURRENT_DIR=$(pwd)
echo "Current directory: '$CURRENT_DIR'"

module load singularity
singularity exec -B /gpfs/projects/bsc70/heka \
                 -B /gpfs/projects/bsc70/heka/repos/tmp_eval_harness/vllm_fix/utils.py:/usr/local/lib/python3.10/dist-packages/vllm/utils.py \
                 -B /gpfs/projects/bsc70/heka/repos/tmp_eval_harness/lm-evaluation-harness/lm_eval/api/task.py:/home/lm-evaluation-harness/lm_eval/api/task.py \
                 -B /gpfs/tapes/MN4/projects/bsc70/hpai/storage/data/heka/Models \
                 -B /gpfs/projects/bsc70/heka/repos/tmp_eval_harness/lm-evaluation-harness/lm_eval/toxigen_generation/toxigen_generation.yaml:/home/lm-evaluation-harness/lm_eval/tasks/toxigen_generation/toxigen_generation.yaml \
                 -B /gpfs/projects/bsc70/heka/repos/tmp_eval_harness/lm-evaluation-harness/lm_eval/toxigen_generation/utils.py:/home/lm-evaluation-harness/lm_eval/tasks/toxigen_generation/utils.py \
                 --nv /gpfs/scratch/bsc70/hpai/storage/projects/heka/singularity/lm_eval_harness042_vllm032_cuda118_new.sif \
   bash -c 'export HF_HUB_OFFLINE=1 && export HF_HOME=/gpfs/scratch/bsc70/hpai/storage/projects/heka/hf_caches/hf_cache2 && export HF_DATASETS_CACHE="/gpfs/scratch/bsc70/hpai/storage/projects/heka/hf_caches/hf_cache2" && \
    python /home/lm-evaluation-harness/lm_eval \
    --model vllm \
    --model_args pretrained='${MODEL_PATH}',tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.8,data_parallel_size=1,max_model_len=8192 \
    --tasks bias_and_toxicity \
    --batch_size auto:4 \
    --num_fewshot 0'
