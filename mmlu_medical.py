import os
import pandas as pandas
import argparse
import json

medical_mmlu_tasks = ["mmlu_high_school_biology", "mmlu_college_biology", "mmlu_college_medicine", 
                    "mmlu_professional_medicine", "mmlu_medical_genetics", "mmlu_virology", 
                    "mmlu_clinical_knowledge", "mmlu_nutrition", "mmlu_anatomy"]


result_path = "../../eval_results/mmlu"

def get_medical_mmlu_score(model_name):
    print(f"opeing {os.path.join(result_path, f'{model_name}.json')}")
    with open(os.path.join(result_path, f"{model_name}.json")) as f:
        results = json.load(f)
    accs = [results['results'][task]['acc,none'] for task in medical_mmlu_tasks]
    print(f"Accuracies: {accs}")
    print(f"Average: {sum(accs)/len(accs)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    

    get_medical_mmlu_score(args.model_name)


if __name__ == "__main__":
    main()

