import os
import pandas as pd
import subprocess
import sys
from io import StringIO
from pathlib import Path

def read_table(file):
    table = pd.read_table(file, sep='|', skiprows=[1]).iloc[:, 1:-1]
    table.columns = table.columns.str.strip()
    table = table.map(lambda x: x.rstrip() if isinstance(x, str) else x)
    return table

def read_tables(file):
    # Read the entire file into a string
    with open(file, 'r') as f:
        content = f.read()

    lines = content.split("\n")

    header_pos = len(lines)
    for i in range(1, len(lines)):
        if "Value  ± Stderr" in lines[i]:
            header_pos = i
    
    first_table = "\n".join(lines[:header_pos])
    if header_pos < len(lines):
        second_table = "\n".join(lines[header_pos+2:])
    
    first_table = read_table(StringIO(first_table))
    if second_table:
        second_table = read_table(StringIO("|      Groups      |    Metric     |Value  ± Stderr|\n|------------------|---------------|-----: ± -----:|\n" + second_table))

    return first_table, second_table

def process_table(task_dict, table, method_column):
    table = table.rename(columns={'Groups': 'Tasks', 'Value  ± Stderr': method_column})

    last_task = None
    for i, row in table.iterrows():
        if type(row['Tasks']) == str and len(row['Tasks']) > 0:
            if "Groups" in row['Tasks']:
                break
            if row['Tasks'] not in task_dict:
                task_dict[row['Tasks']] = {row['Metric']: {method_column: row[method_column]}}
            else:
                if row['Metric'] not in task_dict[row['Tasks']]:
                    task_dict[row['Tasks']][row['Metric']] = {method_column: row[method_column]}
                else:
                    task_dict[row['Tasks']][row['Metric']][method_column] = row[method_column]
            last_task = row['Tasks']
        else:
            if last_task not in task_dict:
                task_dict[last_task] = {row['Metric']: {method_column: row[method_column]}}
            elif type(row['Metric']) == str and len(row['Metric']) > 0:
                if row['Metric'] not in task_dict[last_task]:
                    task_dict[last_task][row['Metric']] = {method_column: row[method_column]}
                else:
                    task_dict[last_task][row['Metric']][method_column] = row[method_column]
    
    return task_dict

def table_to_md(task_dict):
    merged = pd.DataFrame()
    for task, metrics in task_dict.items():
        # Transform the dictionary into a table
        for i, (metric, values) in enumerate(metrics.items()):
            row = pd.DataFrame({'Tasks': ["" if i else task], 'Metric': [metric]})
            for k, v in values.items():
                row[k] = v
            merged = pd.concat([merged, row])
    
    return merged


# Parse the arguments
folder = sys.argv[1]

task_dict = {}
group_dict = {}

# Read the shots folder
for file in os.listdir(folder):
    if file.endswith('.txt'):
        # Get the file name without the extension
        filename = Path(file).stem
        parts = filename.split('_')
        commit, method = parts[0], '_'.join(parts[2:])

        print(f'Processing {filename}...')

        result = subprocess.run(['./format_table.sh', folder, filename], capture_output=True, text=True)

        print(result.stdout.strip())

        first_table, second_table = read_tables(os.path.join(folder, result.stdout.strip()))
        method_column = f'{method} [{commit}]'

        task_dict = process_table(task_dict, first_table, method_column)
        group_dict = process_table(group_dict, second_table, method_column)

output_name = os.path.basename(folder)

task_table = table_to_md(task_dict)
group_table = table_to_md(group_dict)

task_table.to_csv(f'../eval_outs/{output_name}_tasks.csv', index=False)
group_table.to_csv(f'../eval_outs/{output_name}_groups.csv', index=False)

with open(f'../eval_outs/{output_name}.md', 'w+') as file:
    file.write(task_table.to_markdown(index=False).replace('nan', ''))
    file.write("\n")
    file.write(group_table.to_markdown(index=False).replace('nan', ''))
