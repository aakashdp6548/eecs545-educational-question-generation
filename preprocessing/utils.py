'''
Various preprocessing util functions.
'''

import json
import os
import csv
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

def map_passage2questions(source_filename, target_filename):
    '''
    Maps each unique passage to a list of questions from that passage.
    '''
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
    '''
    Saves dictionary mapping passages to lists of questions as a json file
    '''
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
    '''
    Creates a Huggingface dataset from the passage-to-questions data
    '''
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
    '''
    Creates Huggingface dataset from the passage-to-questions files for TedEd data
    '''
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

def write_as_text(dataset, output_dir):
    '''
    Saves a dataset as a text file in the form
    <passage>\t<question>\t<question>
    '''
    splits = [
        ('train', dataset['train']), 
        ('val', dataset['val']),
        ('test', dataset['test'])
    ]

    # create output directory if doesn't exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for name, data in splits:
        output_file = os.path.join(output_dir, f'{name}.txt')
        with open(output_file, 'w') as f:
            tsv_writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for sample in data:
                passage = sample['passage'].strip()
                for question in sample['questions']:
                    question = question.strip()
                    tsv_writer.writerow([passage, question, question])
        print(f'Wrote {output_file}')

def write_as_text_w_answer(output_dir):
    para_file = 'LearningQ/data/experiments/teded/para-test.txt'
    src_file = 'LearningQ/data/experiments/teded/src-test.txt'
    tgt_file = 'LearningQ/data/experiments/teded/tgt-test.txt'
    
    with open(para_file) as para_file, open(src_file) as src_file, open(tgt_file) as tgt_file:
        para_lines = para_file.readlines()
        src_lines = src_file.readlines()
        tgt_lines = tgt_file.readlines()

    para_train, para_test, src_train, src_test, tgt_train, tgt_test = train_test_split(para_lines, src_lines, tgt_lines, test_size=0.2)
    para_val, para_test, src_val, src_test, tgt_val, tgt_test = train_test_split(para_test, src_test, tgt_test, test_size=0.5)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    splits = {
        'train': (para_train, src_train, tgt_train),
        'val': (para_val, src_val, tgt_val),
        'test': (para_test, src_test, tgt_test)
    }
    
    for split, files in splits.items():
        with open(f'{output_dir}/{split}.txt', 'w') as f:
            print(f'Writing {split} file')
            writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for line in zip(files[0], files[1], files[2]):
                writer.writerow([line[0].strip(), line[1].strip(), line[2].strip()])   

def remove_no_ans(input_dir, output_dir):
    # create output directory if doesn't exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        counter = 0
        output_file = os.path.join(output_dir, filename)
        input_file = os.path.join(input_dir, filename)
        print(f'Writing {output_file} from {input_file}')
        with open(input_file, 'r') as in_file, open(output_file, 'w') as out_file:
            writer = csv.writer(out_file, delimiter='\t', quoting=csv.QUOTE_NONE)
            for line in in_file.readlines():
                context, answer, question = line.strip().split('\t')
                context, answer, question = context.strip(), answer.strip(), question.strip()
                if context and answer and question:
                    writer.writerow([context, answer, question])
                else:
                    counter += 1
                    print(line)
        print(f'Removed {counter} lines from {input_file}')


if __name__=='__main__':
    write_as_text_w_answer('data/teded_ans')
