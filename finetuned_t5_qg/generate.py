import pickle
import os
import sys
import tqdm
import torch
import nltk
nltk.download('punkt')

from datasets import load_dataset, load_metric, list_metrics, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollator, T5ForConditionalGeneration, T5TokenizerFast

from tqdm import tqdm

from typing import Dict, List, Optional

import dataclasses
from dataclasses import dataclass, field

import logging
import os
import sys

import argparse

import numpy as np
import torch

from huggingface_hub import notebook_login

from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    EvalPrediction,
    DataCollator,
    Trainer,
    TrainingArguments)

from question_generation import pipeline
from tqdm import trange


def load_local_data_splits(file):
    data = load_from_disk(file)
    try:
        train, val, test = data["train"], data["val"], data["test"]
    except:
        raise Exception("Data does not have train, validation, test splits as keys in storage")
    
    return train, val, test

def produce_pipeline(type, model):
    return (type, pipeline(type, model=model))

def generate_questions(test_data, pipeline):
    gqs = []
    gt = []
    typ = pipeline[0]
    pipeline = pipeline[1]
    contexts, questions = test_data[:]["passage"], test_data[:]["questions"]
    for x in trange(len(contexts)):
        gqs.append(pipeline(contexts[x]))
        gt.append(questions[x])
    
    with open(os.path.join("out", str(typ) + "_outputs.pkl"), "wb") as d:
        pickle.dump(gqs, d)

    with open(os.path.join("out", str(typ) +"_targets.pkl"), "wb") as g:
        pickle.dump(gt, g)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path to hf dataset')
    parser.add_argument('--pipeline-type', type=str, required=True, help='type of pipeline (e2e, multitask-qa-qg)')
    
    args = vars(parser.parse_args())
    return args

def main(args):
    model = 'valhalla/t5-base-e2e-qg'
    train, val, test = load_local_data_splits(args['data'])
    if args['pipeline_type'] == 'multitask-qa-qg':
        model = 'valhalla/t5-base-qa-qg-hl'
    pipeline = produce_pipeline(args['pipeline_type'], model)
    generate_questions(test, pipeline)
    print("Question Generations Complete")

if __name__=='__main__':
    args = get_args()
    main(args)