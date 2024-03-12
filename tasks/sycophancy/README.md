### Dataset
Dataset constructed based on multiple-choice QAs from datasets regarding politics, philosophy and NLP research. 
The aim is to test the extent to which language models are "sycophants" and repeat back user views. They test for 
sycophancy on philosophy questions from the PhilPapers 2020 Survey, Natural Language Processing Survey, and political 
questions from the Pew Research Center Political Typology Quiz. They generate biographies for virtual users who have a 
particular view on the question at hand (for philosophy and Natural Language Processing questions) or who have a 
particular political view (conservative vs. liberal - for political questions). They then append the sourced questions 
to the biography, to simulate a user with a particular view asking a question

Extracted from the paper "Discovering Language Model Behaviors with Model-Written Evaluations" (https://github.com/anthropics/evals/tree/main/sycophancy). 

### Methodology
The data is used to evaluate a model's probability of answering the questions with either " (A)" or " (B)". Having which 
option matches the virtual user behavior, one can compute how often the model's answers matches the view of the person in the biography.

### Results
Accuracy value, 1 meaning completely sycophant.
