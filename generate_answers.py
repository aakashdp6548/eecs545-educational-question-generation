from pipelines import pipeline
import argparse
import os
from datasets import load_from_disk
import csv

def main(args):
    if not os.path.isdir(args['dir_name']):
        os.makedirs(args['dir_name'])

    data = load_from_disk(args['data'])
    qa = pipeline("multitask-qa-qg", model="valhalla/t5-base-qa-qg-hl")

    for split in data:
        file_path = os.path.join(args['dir_name'], f'{split}.txt')
        print(f'Writing {file_path}')
        with open(file_path, 'w') as f:
            writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
        
            for sample in data[split]:
                context = sample['passage'].strip()
                questions = sample['questions']
                for question in questions:
                    question = question.strip()
                    answer = qa({
                        'question': question,
                        'context': context
                    })
                    writer.writerow([context, answer, question])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path to hf dataset containing questions and answers')
    parser.add_argument('--dir_name', type=str, required=True, help='where to save the output')

    args = vars(parser.parse_args())
    return args

if __name__=='__main__':
    args = get_args()
    main(args)