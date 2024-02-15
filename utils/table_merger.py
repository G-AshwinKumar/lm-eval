import math
import pandas as pd
import sys

def read_table(file):
    table = pd.read_table(file, sep='|', skiprows=[1]).iloc[:, 1:-1]
    table.columns = table.columns.str.strip()
    table = table.map(lambda x: x.rstrip() if isinstance(x, str) else x)
    return table


# Parse the arguments
new_method = sys.argv[1]
new_table_path = sys.argv[2]

table = read_table('table.md')

task_dict = {}
last_task = None

# Iterate through the table and create a dictionary of tasks and metrics
for i, row in table.iterrows():
    # If Tasks is not empty add it to the dictionary
    if type(row['Tasks']) == str and len(row['Tasks']) > 0:
        task_dict[row['Tasks']] = {row['Metric']: {c: row[c] for c in table.columns[2:]}}
        last_task = row['Tasks']
    # If Tasks is empty and Metric is not empty, add the metric to the last task
    elif type(row['Metric']) == str and len(row['Metric']) > 0:
        task_dict[last_task][row['Metric']] = {c: row[c] for c in table.columns[2:]}

new_table = read_table(new_table_path).rename(columns={'Value  ± Stderr': new_method})
last_task = None

# Iterate through the new_table adding the new method to the task_dict
for i, row in new_table.iterrows():
    if type(row['Tasks']) == str and len(row['Tasks']) > 0:
        if "Groups" in row['Tasks']:
            break
        if row['Tasks'] not in task_dict:
            task_dict[row['Tasks']] = {row['Metric']: {new_method: row[new_method]}}
        else:
            if row['Metric'] not in task_dict[row['Tasks']]:
                task_dict[row['Tasks']][row['Metric']] = {new_method: row[new_method]}
            else:
                task_dict[row['Tasks']][row['Metric']][new_method] = row[new_method]
        last_task = row['Tasks']
    else:
        if last_task not in task_dict:
            task_dict[last_task] = {row['Metric']: {new_method: row[new_method]}}
        elif type(row['Metric']) == str and len(row['Metric']) > 0:
            if row['Metric'] not in task_dict[last_task]:
                task_dict[last_task][row['Metric']] = {new_method: row[new_method]}
            else:
                task_dict[last_task][row['Metric']][new_method] = row[new_method]

second_table_pos = i + 2

# Create a new table from the dictionary
merged = pd.DataFrame()
for task, metrics in task_dict.items():
    # Transform the dictionary into a table
    for i, (metric, values) in enumerate(metrics.items()):
        row = pd.DataFrame({'Tasks': ["" if i else task], 'Metric': [metric]})
        for k, v in values.items():
            row[k] = v
        merged = pd.concat([merged, row])

# Append second table to the merged table
if second_table_pos < len(new_table):
    merged = pd.concat([merged, new_table.iloc[second_table_pos:]])

with open('merged_table.md', 'w+') as file:
    file.write(merged.to_markdown(index=False).replace('nan', ''))
