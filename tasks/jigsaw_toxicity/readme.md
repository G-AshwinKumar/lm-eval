The dataset we use can be found in: https://huggingface.co/datasets/google/jigsaw_toxicity_pred

The model used to rate toxicity is https://huggingface.co/JiaqiLee/robust-bert-jigsaw.

We add two versions to the task:
* jigsaw_toxicity: evaluates the completion toxicity of the input model on the whole jigsaw_toxicity dataset.
* jigsaw_toxic_prompts: only measures the completion toxicities for toxic prompts in the dataset.

Output ranges from 0 (non-toxic) to 1 (toxic).
