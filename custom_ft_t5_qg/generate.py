import torch

from datasets import load_dataset, load_metric, list_metrics
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollator, T5ForConditionalGeneration, T5TokenizerFast

from tqdm import tqdm

from typing import Dict, List, Optional

import dataclasses
from dataclasses import dataclass, field

import logging
import os
import sys

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

from datasets import load_from_disk
from tqdm import trange
import pickle

import argparse

def load_local_data_splits(file):
    data = load_from_disk(file)
    return data

def prepare_data(all_splits, model, tokenizer):

    tokenizer.sep_token = '<sep>'
    tokenizer.add_tokens(['<sep>'])
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.sep_token_id

    max_input_length =  512
    max_target_length = 64

    def convert_to_features(example_batch):

        input_encodings = tokenizer.batch_encode_plus(example_batch['passage'], 
                                                    max_length=max_input_length, 
                                                    add_special_tokens=True,
                                                    truncation=True, 
                                                    pad_to_max_length=True)
        
        target_encodings = tokenizer.batch_encode_plus(example_batch['questions'], 
                                                    max_length=max_target_length, 
                                                    add_special_tokens=True,
                                                    truncation=True, pad_to_max_length=True)
                                                    
        encodings = {
            'input_ids': input_encodings['input_ids'], 
            'attention_mask': input_encodings['attention_mask'],
            'decoder_input_ids': target_encodings['input_ids']
            ,'decoder_attention_mask': target_encodings['attention_mask']
        }

        return encodings

    def flatten_list(inp):
        inp["questions"] = ' {sep_token} '.join(inp["questions"])
        inp["questions"] = inp["questions"].strip()
        return inp

    def add_gen(inp):
        inp["passage"] = 'generate questions: ' + inp["passage"].strip()
        return inp

    def add_eos_examples(example):
        example['passage'] = example['passage'] + " </s>"
        example['questions'] = example['questions'] + " </s>"
        return example

    def add_special_tokens(example):
        example['questions'] = example['questions'].replace("{sep_token}", '<sep>')
        return example


    all_splits = all_splits.map(flatten_list)
    all_splits = all_splits.map(add_gen)
    tokenized_dataset  = all_splits.map(add_eos_examples)
    tokenized_dataset = tokenized_dataset.map(add_special_tokens)
    tokenized_dataset  = tokenized_dataset.map(convert_to_features,  batched=True)
    train_data, val_data, test_data = tokenized_dataset["train"], tokenized_dataset["val"], tokenized_dataset["test"]
    columns = ['input_ids', 'decoder_input_ids', 'attention_mask', 'decoder_attention_mask']
    train_data.set_format(type='torch', columns=columns)
    val_data.set_format(type='torch', columns=columns)
    test_data.set_format(type='torch', columns=columns)
    torch.save(train_data, 'train_data.pt')
    torch.save(val_data, 'val_data.pt')
    torch.save(test_data, 'test_data.pt')
    return train_data, val_data, test_data, model, tokenizer

# This dataclass implementation is taken from Suraj Patil: https://github.com/patil-suraj/question_generation
@dataclass
class T2TDataCollator():
  def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
    A dictionary of tensors
    """
    
    input_ids = torch.stack([example['input_ids'] for example in batch])
    lm_labels = torch.stack([example['decoder_input_ids'] for example in batch])
    lm_labels[lm_labels[:, :] == 0] = -100 
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    decoder_attention_mask = torch.stack([example['decoder_attention_mask'] for example in batch])
    
    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask,
        'labels': lm_labels, 
        'decoder_attention_mask': decoder_attention_mask
    }

def train(train_data, val_data, model):

    training_args = TrainingArguments(output_dir="models", 
                                    per_device_train_batch_size=4, 
                                    per_device_eval_batch_size=4,
                                    gradient_accumulation_steps=16,
                                    learning_rate=1e-4, 
                                    num_train_epochs=150,
                                    logging_steps=100,
                                    run_name="end2end-questions-generation",
                                    evaluation_strategy="steps",
                                    save_steps=500)

    logger = logging.getLogger(__name__)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=T2TDataCollator()
    )

    # Training
    trainer.train()

def generate_questions(tokenizer, model, test_data):

# model = T5ForConditionalGeneration.from_pretrained("./models/checkpoint-1000/")

    def hf_run_model(input_string, **generator_args):
        generator_args = {
        "max_length": 256,
        "num_beams": 4,
        "length_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        }
        input_string = "generate questions: " + input_string + " </s>"
        input_ids = tokenizer.encode(input_string, return_tensors="pt").to("cuda")
        res = model.generate(input_ids, **generator_args)
        output = tokenizer.batch_decode(res, skip_special_tokens=True)
        output = [item.split("<sep>") for item in output]
        return output

    outputs, gt = [], []

    for i in trange(len(test_data)):
        entry = test_data[i]
        outputs.append(hf_run_model(entry["passage"]))
        gt.append(entry["questions"])

    with open(os.path.join("out", "finetune_outputs.pkl"), 'wb') as f:
        pickle.dump(outputs, f)
    with open (os.path.join("out","finetune_targets.pkl"), 'wb') as g:
        pickle.dump(gt, g)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path to hf dataset')
    parser.add_argument('--model', type=str, required=True, help='type of model (t5-small, t5-base)')
    parser.add_argument('--load-model', type='str', default='')
    
    args = vars(parser.parse_args())
    return args

def main(args):
    # checkpoint = "t5-base"]
    checkpoint = args['model']
    tokenizer = T5TokenizerFast.from_pretrained(checkpoint)

    if args['load-model'] == '':
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args['load-model'])
    
    all_splits = load_local_data_splits(args['data'])

    train_data, val_data, test_data, model, tokenizer = prepare_data(all_splits, model, tokenizer)
    train(train_data, val_data, model)
    # use the below only if loading from local storage, change checkpoint to the relevant one
    # model = T5ForConditionalGeneration.from_pretrained("./models/checkpoint-1000/")
    generate_questions(tokenizer, model, test_data)

    print("Question Generations Complete")

if __name__=='__main__':
    args = get_args()
    main(args)