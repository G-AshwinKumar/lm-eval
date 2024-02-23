#!/bin/bash

# Check if three arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 path filename"
    exit 1
fi

# Extract arguments
path="./$1"
filename="$2"

# Navigate to the specified directory
cd "$path" || exit 1

# Find the .o file and extract the table
table=$(grep -E "^\|" ${filename}.txt | awk -F'|' '/Tasks/ {p=1} p')

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
output_file="${filename}_formatted.o"

echo "$reduced_table" >| "$output_file"
echo "$new_row" >> "$output_file"

echo $output_file
