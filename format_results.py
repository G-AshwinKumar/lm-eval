import argparse
import json
import pandas as pd
from pathlib import Path
import os

def get_acc(acc_dict):
    return list(acc_dict.values())[0]

def read_json_files(json_dir, model_name):
    # Get a list of all JSON files in the directory
    json_files = list(Path(json_dir).rglob(f"*{model_name}.json"))

    # Initialize an empty list to store accuracy results
    results_list = []

    # Iterate through each JSON file
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

            # Extract accuracy results
            task_alias = list(data['results'].keys())[0]

            accuracy_result = data['results'][task_alias]

            # Append results to the list
            results_list.append({'Task': task_alias, 'Accuracy': accuracy_result, 'Model': model_name})

    # Create a DataFrame from the results list
    df = pd.DataFrame(results_list)

    return df

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Process JSON files and calculate accuracy.')

    # Add an argument for the model name
    parser.add_argument('--model', type=str, help='Name of the model')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Define the directory containing your JSON files
    json_dir = "../../eval_results"

    # Call the function with the specified directory and model name
    result_df = read_json_files(json_dir, args.model)

    # use get_acc
    result_df['avg'] = result_df['Accuracy'].apply(get_acc)

    # add mean column
    avg_acc = result_df['avg'].mean()

    result_df = result_df.append({'Task': 'mean', 'Accuracy': avg_acc, 'Model': args.model}, ignore_index=True)

    print(result_df)

    # Save the DataFrame as a CSV file
    save_path = os.path.join("../../eval_results", f"{args.model}_results.csv")

    if not os.path.exists(save_path):

        result_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()
