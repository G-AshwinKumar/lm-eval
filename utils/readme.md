# lm_evaluation_harness/utils

## table_merger.py

This script can be used to generate a markdown table combining the result of different model evaluations.

To launch it, use `python table_merger.py <folder>` where `<folder>` is the path where the evaluation results are stored.

The script will create a table titled `<folder>` and will automatically call `format_table.sh`.