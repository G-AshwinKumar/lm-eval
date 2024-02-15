#!/bin/bash

# Check if three arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 commit_tag cesga_user job_pid"
    exit 1
fi

# Extract arguments
commit_tag="$1"
cesga_user="$2"
job_pid="$3"

path=""
output_file="${path}formatted_out.o"

# Construct the full path
full_path="${path}job_${commit_tag}.o"

# Check if the .o file exists
if [ ! -f "$full_path" ]; then
    echo "Directory does not exist: $full_path"
    exit 1
fi

# Navigate to the specified directory
cd "$path" || exit 1

# Find the .o file and extract the table
table=$(grep -E "^\|" *.o | awk -F'|' '/Tasks/ {p=1} p')

# Process the table and create the reduced version
reduced_table=$(echo "$table" | awk -F'|' 'BEGIN{OFS="|"} {if (NR==2) {printf "|%s|%s|%s---%s|\n", $2, $6, $7, $9} else {printf "|%s|%s|%s ± %s|\n", $2, $6, $7, $9}}')

# Extract the rows corresponding to specific categories
selected_rows=$(echo "$reduced_table" | grep -E "college_biology|professional_medicine|virology|clinical_knowledge|college_medicine|anatomy")

# Extract the mean and stderr of [college_biology|professional_medicine|virology|clinical_knowledge|college_medicine|anatomy]
LC_NUMERIC=C mean=$(echo "$selected_rows" | awk -F' ± ' '{gsub(/,/, ".", $1); split($1, arr, "|"); sum += arr[length(arr)];} END {printf "%.4f", sum / NR}')
LC_NUMERIC=C stderr=$(echo "$selected_rows" | awk -F' ± ' '{gsub(/,/, ".", $2); sum += $2} END {printf "%.4f", sum / NR}')

# Create a new row
new_row="| - medicine       |acc            |$mean ± $stderr|"

# Write the reduced table to the output file
echo "$reduced_table" >| "$output_file"
echo "$new_row" >> "$output_file"

echo $output_file
