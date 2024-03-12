## Dataset
For this benchmark we use ETHOS, a hate speech detection dataset built from YouTube and Reddit comments validated 
through a crowdsourcing platform. It has two subsets, one for binary classification and the other for multi-label 
classification. The former contains 998 comments, while the latter contains fine-grained hate-speech annotations for 433 comments.

## Methodology
The methodology followed is the same to the one followed in ToxiGen:

1. Prompt the LLM with the ETHOS text instances, which it will extend.
2. Use a hate-detection pretrained ToxDectRoBERTa to assess how toxic the LLM generations are.
3. Aggregate the scores

[ethos_generation_general]
This is computed for the whole 'binary' set, which contains around 1K instances.

[ethos_generation_{population}]
This is computed for a population-specific set of instances, within the 'multilabel' set.

## Results
A toxic coefficient of 1 will mean that our model is very toxic.
A toxic coefficient of 0 will mean our model is not toxic at all.