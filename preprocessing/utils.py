'''
Various preprocessing util functions.
'''

import numpy
import os
import json

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