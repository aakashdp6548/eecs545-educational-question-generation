'''
Various preprocessing util functions.
'''

import json
import os
from datasets import load_dataset, DatasetDict

def map_passage2questions(source_filename, target_filename):
    with open(source_filename) as f:
        passage_lines = f.readlines()    

    with open(target_filename) as f:
        question_lines = f.readlines()

    data = {}
    for passage, question in zip(passage_lines, question_lines):
        if passage not in data:
            data[passage] = []
        data[passage].append(question)
    
    return data

def save_passage2questions(source_filename, target_filename, output_filename):
    data_dir = map_passage2questions(source_filename, target_filename)

    output = []
    for passage in data_dir.keys():
        output.append({
            'passage': passage,
            'questions': data_dir[passage]
        })
    
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)


def create_dataset():
    data_files = {
        'train': 'test_data/passage-to-questions/p2q-train.json',
        'dev': 'test_data/passage-to-questions/p2q-dev.json',
        'test': 'test_data/passage-to-questions/p2q-test.json'
    }
    dataset = load_dataset('json', data_files=data_files)
    return dataset

def process_teded_data():
    data_dir = 'LearningQ/data/experiments/teded/'
    output_dir = 'test_data/teded_passage-to-questions/'

    para_file = os.path.join(data_dir, 'para-test.txt')
    src_file = os.path.join(data_dir, 'src-test.txt')
    tgt_file = os.path.join(data_dir, 'tgt-test.txt')

    save_passage2questions(para_file, tgt_file, os.path.join(output_dir, 'para_tgt_test.json'))
    save_passage2questions(src_file, tgt_file, os.path.join(output_dir, 'src_tgt_test.json'))


def create_teded_dataset():
    para_tgt_file = 'test_data/teded_passage-to-questions/para_tgt_test.json'
    src_tgt_file = 'test_data/teded_passage-to-questions/src_tgt_test.json'

    for data_type, filename in zip(['para', 'src'], [para_tgt_file, src_tgt_file]):
        dataset = load_dataset('json', data_files=filename)
        train_testval = dataset['train'].train_test_split(test_size=0.2) # reserve 20% for val/test
        test_val = train_testval['test'].train_test_split(test_size=0.5) # 50% of reserved 20%
        new_dataset = DatasetDict({
            'train': train_testval['train'],
            'test': test_val['test'],
            'val': test_val['train']
        })
        new_dataset.save_to_disk(f'test_data/teded_full_{data_type}.hf')


if __name__=='__main__':
    create_teded_dataset()
