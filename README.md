# eecs545-educational-question-generation

Midterm Progress Report Planning
1. Preprocess data
- https://huggingface.co/mrm8488/t5-base-finetuned-question-generation-ap
- For each question, compute similarity with passage to extract most relevant sentences, then case, tokenize, split
2. Modify script (https://colab.research.google.com/github/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb#scrollTo=ptPupnLsfkMH) to work with GPU, get this to work on Great Lakes or Colab
3. Finetune the model and test, adjust as needed


Other Notes
- focusing on single-hop questions for now, consider multi-hop later
