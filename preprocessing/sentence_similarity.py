'''
Script to extract sentences from context passage based on similarity
to the question. Currently only the most relevant sentence based on
the similarity metric is chosen, might update later to select 2+
sentence.

Usage: (from root directory)
    ./preprocesing/sentence_similarity.py <SOURCE_FILE> <QUESTIONS_FILE> <OUTPUT_FILE> [SAMPLE_SIZE]
    Run "./preprocessing/sentence_similarity.py -h" for more details. 
'''

import argparse
import json
import os

import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

DATA_DIR='LearningQ/data/experiments/khan/'
SENTENCE_FILE = 'src-train.txt'
QUESTIONS_FILE = 'tgt-train.txt'

SAMPLE_SIZE = 10

def load_data(source_filename, target_filename, sample_size=None):
    '''
    Loads the data from the files and extracts random sample (if desired).
    Parameters:
        source_filename: the source file (from CLI)
        target_filename: the questions file (from CLI)
        sample_size: the number of source/question pairs to randomly select
                       (optional, returns all by default)
    Returns:
        Tuple of sentence and question lists
    '''
    # get context and target files
    source_file = os.path.join(DATA_DIR, source_filename)
    questions_file = os.path.join(DATA_DIR, target_filename)

    with open(source_file) as f:
        sentence_lines = f.readlines()    

    with open(questions_file) as f:
        question_lines = f.readlines()

    if len(sentence_lines) != len(question_lines):
        raise Exception("Number of lines in source and target files don't match")

    # Pull <SAMPLE_SIZE> sentences / questions randomly
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(sentence_lines):
        data = list(zip(sentence_lines, question_lines))
        np.random.shuffle(data)
        sample = data[0:SAMPLE_SIZE]
        sentence_lines, question_lines = zip(*sample)

    return sentence_lines, question_lines

def tokenize_sentences(sentence_arr):
    '''
    Splits each source into sentences.
    Parameters:
        sentence_arr: list of sentence lines from source file
    Returns:
        List of lists of split sentences per line
    '''
    return list(map(sent_tokenize, sentence_arr))

def get_top1_sentence(question, sentences, model):
    '''
    Computes similarity for a single question.
    Parameters:
        questions: question to compute similarity for
        sentences: list of list of sentences to compute similarity to
        model: SentenceTransformer model to use to get embedding vectors
    Returns:
        Tuple of sentence, similarity_score
    '''
    # Convert question and sentence vectors to embeddings
    q_vec = model.encode(question, convert_to_tensor=True)
    sent_vecs = [model.encode(sentence, convert_to_tensor=True) for sentence in sentences]

    # Compute cosine similarity and get most similar sentence
    similarity = [util.pytorch_cos_sim(q_vec, sent_vec).item() for sent_vec in sent_vecs]
    return sentences[np.argmax(similarity)], np.max(similarity)

def main(args):
    '''
    Loads data, computes similarity, and saves sentences to output file.
    Parameters:
        args: command line parameters
    '''
    # Load data and tokenize sentences
    sentences, questions = load_data(args.source, args.target, args.sample_size)
    tokenized_sentences = tokenize_sentences(sentences)

    # Compute similarity for each question to its sentences
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    output = []
    for question, sentences in zip(questions, tokenized_sentences):
        top1_sentence, similarity = get_top1_sentence(question, sentences, model)
        output.append({
            'question': question,
            'sentences': sentences,
            'top1_sentence': top1_sentence,
            'similarity': similarity
        })

    # Write new sentences to file
    with open(args.write, 'w') as f:
        json.dump(output, f, indent=2)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='source file (e.g. "src-train.txt")')
    parser.add_argument('--target', type=str, required=True, help='questions file (e.g. "tgt-train.txt")')
    parser.add_argument('--write', type=str, required=True, help='file to write new sentences to (full path)')
    parser.add_argument('--sample_size', type=int, required=False, default=None, 
                        help='number of sentences to load (optional; selected randomly from source/question files)')
    args = parser.parse_args()

    main(args)