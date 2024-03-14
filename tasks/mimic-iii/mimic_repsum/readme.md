## Mimic-III Report summarization

The dataset can be found in: https://huggingface.co/datasets/dmacres/mimiciii-hospitalcourse-meta

We are evaluating the following metrics:

* **BERT**: "an automatic evaluation metric for text generation that computes a similarity score for each token in the candidate sentence with each token in the reference sentence. It leverages the pre-trained contextual embeddings from BERT models and matches words in candidate and reference sentences by cosine similarity."

* **ROUGE**: "or Recall-Oriented Understudy for Gisting Evaluation compares an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation." We report "rouge1": unigram (1-gram) based scoring,
"rouge2": bigram (2-gram) based scoring and "rougeL": Longest common subsequence based scoring.

* **RadGraph F1**: RadGraph is a model trained on RadGraph, a dataset of entities and relations in full-text radiology reports and capable of comparing entities and relations in the candidate sentence present in the reference sentence. 


### The RadGraph F1 score.

The original radgraph model (https://physionet.org/content/radgraph/1.0.0) depends on allenNLP, which is no longer maintained.

We are following the implementation in https://arxiv.org/abs/2210.12186.
Also: https://github.com/davevanveen/radadapt/tree/main

