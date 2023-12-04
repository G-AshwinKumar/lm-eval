import argparse
import json
import pandas as pd
from pathlib import Path
import os

metrics_map_original = {'arc_challenge': 'acc-norm','hellaswag': 'acc-norm',
    'truthfulqa_mc1': 'mc2','mmlu': 'acc','winogrande': 'acc','gsm8k': 'acc','drop': 'f1'
}
metrics_map = {
    'arc_challenge': 'acc_norm,none',
    'hellaswag': 'acc_norm,none',
    'truthfulqa_mc1': 'acc,none',
    'mmlu': 'acc,none',
    'winogrande': 'acc,none',
    'gsm8k': 'exact_match,get-answer',
    'drop': 'f1,none',
    'mmlu_medical': 'acc,none',
    'pubmedqa': 'acc,none'
}

medical_mmlu_tasks = ["mmlu_high_school_biology", "mmlu_college_biology", "mmlu_college_medicine", 
                    "mmlu_professional_medicine", "mmlu_medical_genetics", "mmlu_virology", 
                    "mmlu_clinical_knowledge", "mmlu_nutrition", "mmlu_anatomy"]


def get_acc(acc_dict):
    if isinstance(acc_dict, dict):
        return acc_dict[metrics_map[acc_dict['alias']]]   
    else:
        raise ValueError(f"acc_dict is not a dict: {acc_dict}")

def read_json_files(json_dir, model_name):
    # Get a list of all JSON files in the directory
    json_files = list(Path(json_dir).rglob(f"*{model_name}.json"))

    # Initialize an empty list to store accuracy results
    results_list = []

    # Iterate through each JSON file
    for json_file in json_files:

        with open(json_file, 'r') as f:
            data = json.load(f)

            print(f"Loaded {json_file}")

            # Extract accuracy results
            task_alias = list(data['results'].keys())[0]

            accuracy_result = data['results'][task_alias]

             
            if task_alias == 'mmlu' or "mmlu_medical" in json_file.parent.name:
                mmlu_medical_results = {task: data['results'][task]['acc,none'] for task in medical_mmlu_tasks}
                print(f"MMlu Medical Results: {mmlu_medical_results}")
                mmlu_medical_avg = sum(mmlu_medical_results.values())/len(mmlu_medical_results)
                results_list.append({'Task': 'mmlu_medical', 'Model': model_name, 'Target_metric': metrics_map["mmlu"], 'acc,none': mmlu_medical_avg})
            
            if "mmlu_medical" in json_file.parent.name:
                continue

            # Append results to the list
            results_list.append({'Task': task_alias, 'Model': model_name, 'Target_metric': metrics_map[task_alias], **accuracy_result})

           

    # Create a DataFrame from the results list
    df = pd.DataFrame(results_list)

    return df

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Process JSON files and calculate accuracy.')

    parser.add_argument('--model', type=str, help='Name of the model', required=True)

    args = parser.parse_args()

    json_dir = "../../eval_results"

    result_df = read_json_files(json_dir, args.model)

    result_df.insert(3, 'benchmark', result_df.apply(lambda row: row[row['Target_metric']], axis=1))
        
    hf_leaderboard = (result_df['Task'] == 'mmlu_medical') | (result_df['Task'] == 'pubmedqa')


    mean_benchmark = result_df.loc[~hf_leaderboard, 'benchmark'].mean()

    result_df = result_df.append({'Task': 'Open LLM', 'Model': args.model, 'Target_metric': 'avg', 'benchmark': mean_benchmark}, ignore_index=True)

    # result_df = result_df.append({'Task': 'mean', 'Model': args.model, 'Target_metric': 'avg', 'benchmark': result_df['benchmark'].mean()}, ignore_index=True)
    result_df = result_df.drop(columns=['alias'])
   
    save_path = os.path.join("../../eval_results", f"{args.model}_results.csv")

    result_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()
