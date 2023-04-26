# Answer-Agnostic Question Generation for Educational Questions

This repository contains the code for the EECS 545 W23 project "Answer-Agnostic Question Generation for Educational Questions". In this project, we aim to investigate question generation models that do not require an answer at train time, hence "answer-agnostic".

We experiment with three different models:
1. T5, a transformer-based model
2. Graph2Seq, a graph neural network
3. ChatGPT

## Data
We use the LearningQ dataset, which consists of educational questions from Khan Academy and TedEd. The raw data can be found on the LearningQ github repo. We include some preprocessing data in ``data``, and functions used to preprocess the raw data in ``preprocessing/utils.py``.

## T5
The code for the T5 models can be found under the subfolders ``custom_ft_t5_qg``, which contains the multi-task and end-to-end inference models, and ``finetuned_t5_qg``, which contains the finetuning code.

## Graph2Seq
The code for the Graph2Seq model can be found under the ``g2s_question_generation`` subfolder. The code requires CoreNLP to be installed and running. ``main.py`` contains the main training code, and ``inference_advance.py`` contains the code for evaluation and inference.

## ChatGPT
The code for the summarizer + ChatGPT is under ``summarizer_chatgpt``. 

## Evaluation
The code for calculating the BLEU, ROUGE, and METEOR metrics is in ``evaluate.py``.