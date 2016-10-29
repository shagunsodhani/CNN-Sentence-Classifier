# CNN-Sentence-Classifier
Simplified implementation of "Convolutional Neural Networks for Sentence Classification" paper

## Usage

* Install [Keras](https://keras.io/#installation)
* Repository contains "Movie reviews with one sentence per review" (Pang and Lee, 2005) dataset in `sample_dataset`.
* Alternatively, to use some other dataset, make two files
    * `input.txt` where each line is a sentence to be classified
    * `label.txt` where each line is the label for corresponding line in `input.txt`
* Make `model` folder by running `mkdir model`
* Refer [this](http://nlp.stanford.edu/projects/glove/) to train or download Glove embeddings and [this](https://code.google.com/archive/p/word2vec/) for Word2Vec embeddings.
* Run `python3 app/train.py --data_dir=path_to_folder_containing_input.txt_and_label.txt --embedding_file_path=path_to_embedding_vectors_file --model_name=name_of_model_from_the_paper`
* For example, if data is in `data` folder, embedding file is `vectors.txt` and model is `cnn_static`, run `python3 app/train.py --data_dir=data --embedding_file_path=vectors.txt --model_name=cnn_static`
* To define your own model, pass `model_name` as `self`, define your model in [app/model/model.py](app/model/model.py) and invoke from `model_selector` function (in [model.py](app/model/model.py)).
* All supported arguments can be seen in [here](app/utils/argumentparser.py)

## References

* [Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014)](https://arxiv.org/abs/1408.5882)
