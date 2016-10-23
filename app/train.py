from __future__ import print_function

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from model.model import model_selector
from reader.filereader import read_glove_vectors, read_input_data
from utils import argumentparser

np.random.seed(42)


def main():
    args = argumentparser.ArgumentParser()
    train(args)


def train(args):
    print('Reading word vectors.')
    embeddings_index = read_glove_vectors(args.embedding_file_path)
    print('Found {} word vectors.'.format(len(embeddings_index)))

    # remote = callbacks.RemoteMonitor(root='http://localhost:9000')

    print('Processing input data')
    texts, labels_index, labels = read_input_data(args.data_dir)
    # texts - list of text samples
    # labels_index - dictionary mapping label name to numeric id
    # labels - list of label ids
    print('Found {} texts.'.format(len(texts)))

    # Vectorize the text sample into 2D integer tensor
    tokenizer = Tokenizer(nb_words=args.nb_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found {} unique tokens.'.format(len(word_index)))

    data = pad_sequences(sequences, maxlen=args.max_sequence_len)

    # Transform labels to be categorical variables
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the input data into training set and validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(args.validation_split * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print('Preparing embedding matrix.')

    # initiate embedding matrix with zero vectors.
    nb_words = min(args.nb_words, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, args.embedding_dim))
    for word, i in word_index.items():
        if i > nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    args.nb_words = nb_words
    args.len_labels_index = len(labels_index)

    model = model_selector(args, embedding_matrix)
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              nb_epoch=args.num_epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
