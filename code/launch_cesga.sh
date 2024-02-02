#!/bin/bash
#------------------------------------------------------
# Example SLURM job script with SBATCH requesting GPUs
#------------------------------------------------------
#SBATCH -J test-cesga       # Job name
#SBATCH --gres=gpu:a100:1   # Request 1 GPU of 2 available on an average A100 node
#SBATCH -c 32               # Cores per task requested
#SBATCH -t 00:05:00         # Run time (hh:mm:ss) - 30 min
#SBATCH --mem=120G        # Memory per node

module load singularity/3.9.7
singularity exec -B /mnt  --nv /mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/Singularity/lm_eval_harness_vllm_cuda118.sif bash -c "python test.py $1 $2"

