import pandas as pd
import string
import evaluate

def meteor_scores_qc(outputs, targets):
    meteor = evaluate.load("meteor")
    meteor_scores_q = [[meteor.compute(predictions = [outputs[i][j]], references=[targets[i][j]]) 
                            for j in range(len(outputs[i]))] for i in range(len(outputs))]
    meteor_scores_c = [meteor.compute(predictions = outputs[i], references=targets[i]) 
                            for i in range(len(outputs))]
    

def meteor_score(outputs_flat, targets_flat):
    meteor = evaluate.load("meteor")
    return (meteor.compute(predictions = outputs_flat, references=targets_flat))

def rouge_scores_qc(outputs, targets):
    rouge = evaluate.load('rouge')
    rouge_scores_q = [[rouge.compute(predictions = [outputs[i][j]], references=[targets[i][j]]) 
                          for j in range(len(outputs[i]))] for i in range(len(outputs))]
    rouge_scores_c = [rouge.compute(predictions = outputs[i], references=targets[i]) 
                          for i in range(len(outputs))]
    
def rouge_score(outputs_flat, targets_flat):
    rouge = evaluate.load('rouge')
    return(rouge.compute(predictions = outputs_flat, references=targets_flat))

def bleu_scores_qc(outputs, targets, n):
    bleu = evaluate.load("bleu")
    bleu_scores_q = [[bleu.compute(predictions = [outputs[i][j]], references=[targets[i][j]], max_order = n) 
                          for j in range(len(outputs[i]))] for i in range(len(outputs))]
    bleu_scores_c = [bleu.compute(predictions = outputs[i], references=targets[i], max_order = n) 
                     for i in range(len(outputs))]
    
def bleu_score(outputs_flat, targets_flat, n):
    bleu = evaluate.load("bleu")
    return(bleu.compute(predictions=outputs_flat, references=targets_flat, max_order = n)["bleu"])

def metrics_all(outputs_flat, targets_flat):
    meteor = meteor_score(outputs_flat, targets_flat)
    rouge = rouge_score(outputs_flat, targets_flat)
    all_scores = {}
    for n in range(1, 5):
        all_scores[f"bleu{n}"] = (bleu_score(outputs_flat, targets_flat, n))
    all_scores["meteor"] = meteor
    all_scores["rouge"] = rouge
    return all_scores

def metrics(outputs, targets):
    no_punc =str.maketrans('','',string.punctuation)
    qs = None
    if isinstance(outputs[0][0], dict):
        qs = [[qa['question'].translate(no_punc).lower() for qa in outputs[i]] for i in range(len(outputs))]
    else:
        qs = [[outputs[i][j].translate(no_punc).lower() for j in range(len(outputs[i]))] for i in range(len(outputs))]
    targets = [[targets[i][j].translate(no_punc).lower() for j in range(len(targets[i]))] for i in range(len(targets))]
    dup_targets = [[targets[i] for j in range(len(qs[i]))] for i in range(len(qs))]
    dup_targets_flat = [targets[i] for i in range(len(qs)) for j in range(len(qs[i]))]
    qs_flat = [qs[i][j] for i in range(len(qs)) for j in range(len(qs[i]))]
    return(metrics_all(qs_flat, dup_targets_flat))

