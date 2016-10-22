from __future__ import print_function

import argparse


def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data directory containing input.txt and label.txt')
    parser.add_argument('--glove_file_path', type=str, default='vectors.txt',
                        help='path to file for glove vectors')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='directory to store checkpointed models')
    parser.add_argument('--nb_words', type=int, default=20000,
                        help='Number of words to keep from the dataset')
    parser.add_argument('--max_sequence_len', type=int, default=56,
                        help='Maximum input sequence length')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='Fraction of data to be used for validation')
    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='Dimension of the embedding space to be used')
    parser.add_argument('--model_name', type=str, default='cnn-rand',
                        help='Name of the model variant, from the CNN Sentence '
                             'Classifier paper. Possible values are cnn-rand, cnn-static'
                             'cnn-non-static. If nothing is specified, it uses the arguments'
                             'passed to the script to define the hyperparameters. To add'
                             'your own model, pass model_name as self, define your model in'
                             'app/model/model.py and invoke from model_selector function.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--save_frequency', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--device', type=str, default='/cpu:0',
                        help='Computing device to use for training. \
                        \'/cpu:0\' to use CPU of the machine.\
                        \'/gpu:0\' to use the first GPU of the machine (if there is a GPU).\
                        \'/gpu:1\' to use the second GPU of the machine and so on.')
    return parser.parse_args()
