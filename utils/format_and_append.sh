#!/bin/bash

# Check if three arguments are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 commit_tag cesga_user job_pid method_name"
    exit 1
fi

# Extract arguments
commit_tag="$1"
cesga_user="$2"
job_pid="$3"
method_name="$4"

out_file="/mnt/lustre/scratch/nlsas/home/res/cns10/SHARE/pipelines/lm-evaluation-harness/$commit_tag/slurm_output/out.txt"

let "a = 0"
while [[ "$(squeue | grep $job_pid)" != "" ]]
do
  if [[ $((a % 5)) == 0 ]]; then
    squeue -l
    squeue -l --start
  fi

  cat $out_file | tail -n 15
  echo "Still running..."
  sleep 60
  let "a = a + 1"
done

# Once the loop finishes, call format_table.sh
output_path=$(./format_table.sh $commit_tag $cesga_user $job_pid)

# Check if format_table.sh was successful
if [ $? -ne 0 ]; then
    echo "format_table.sh failed with exit code $?"
    exit $?
fi

# If format_table.sh was successful, call table_merger.py
python3 table_merger.py $method_name $output_path
