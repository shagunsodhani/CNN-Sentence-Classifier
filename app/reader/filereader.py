from __future__ import division
from __future__ import print_function

import os

import numpy as np


def read_glove_vectors(glove_vector_path):
    '''Method to read glove vectors and return an embedding dict.'''
    embeddings_index = {}
    with open(glove_vector_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs[:]
    return embeddings_index


def read_input_data(input_data_path):
    '''Method to read data from input_data_path'''
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    texts = list(open(os.path.join(input_data_path, "input.txt"), "r").readlines())

    with open(os.path.join(input_data_path, "label.txt"), 'r') as label_f:
        largest_label_id = 0
        for line in label_f:
            label = str(line.strip())
            if label not in labels_index:
                labels_index[label] = largest_label_id
                largest_label_id += 1
            labels.append(labels_index[label])

    return texts, labels_index, labels
