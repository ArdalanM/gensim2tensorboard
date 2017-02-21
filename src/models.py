# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan77400@gmail.com>
@brief:
"""

import os
import csv
import gensim
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


class CsvConnector(object):
    def __init__(self, filepath=None, separator=',', columns_to_select=(), columns_joining_token='. ',
                 preprocessing=None):

        if not columns_to_select:
            print("You have to select at least one column on your input data")
            raise

        with open(filepath, 'r', encoding='utf-8') as f:
            self.reader = csv.DictReader(f, delimiter=separator, quotechar='"')
            columns = self.reader.fieldnames
            for col in columns_to_select:
                if col not in columns:
                    print("{} is not a valid column. Found {}".format(col, columns))
                    raise

        if not preprocessing:
            preprocessing = lambda x: x

        self.filepath = filepath
        self.separator = separator
        self.columns_to_select = columns_to_select
        self.columns_joining_token = columns_joining_token
        self.preprocessing = preprocessing

    def __iter__(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=self.separator, quotechar='"')
            for line in reader:
                sentence = self.columns_joining_token.join([line[col] for col in self.columns_to_select])
                yield self.preprocessing(sentence).split()


class TxtConnector(object):
    def __init__(self, filepath=None, preprocessing=None):

        if not preprocessing:
            preprocessing = lambda x: x

        self.filepath = filepath
        self.preprocessing = preprocessing

    def __iter__(self):
        for line in open(self.filepath, 'r', encoding='utf-8'):
            yield self.preprocessing(line).split()


class Bigram(object):

    def __init__(self, iterator):
        self.iterator = iterator
        self.bigram = gensim.models.Phrases(self.iterator)

    def __iter__(self):
        for sentence in self.iterator:
            yield self.bigram[sentence]


class Word2Vec(object):
    def __init__(self, model=None, save_folder=None, phrases=False):
        if not os.path.exists(save_folder):
            print("{} Folder does not exist, create it first".format(save_folder))

        self.model = model
        self.save_folder = save_folder
        self.phrases = phrases

    def fit(self, sentences, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None,
            sample=1e-3, seed=1, workers=4, min_alpha=0.0001, sg=0, hs=0, negative=10,
            cbow_mean=1, iter=5, null_word=0):
        self.model = gensim.models.Word2Vec(sentences,
                                            size=size,
                                            alpha=alpha,
                                            window=window,
                                            min_count=min_count,
                                            max_vocab_size=max_vocab_size,
                                            sample=sample,
                                            seed=seed,
                                            workers=workers,
                                            min_alpha=min_alpha,
                                            sg=sg,
                                            hs=hs,
                                            negative=negative,
                                            cbow_mean=cbow_mean,
                                            iter=iter,
                                            null_word=null_word)

        self.model.save(os.path.join(self.save_folder, "gensim-model.cpkt"))


def create_embeddings(gensim_model=None, model_folder=None):
    weights = gensim_model.wv.syn0
    idx2words = gensim_model.wv.index2word

    vocab_size = weights.shape[0]
    embedding_dim = weights.shape[1]

    with open(os.path.join(model_folder, "metadata.tsv"), 'w') as f:
        f.writelines("\n".join(idx2words))

    tf.reset_default_graph()

    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)

    writer = tf.summary.FileWriter(model_folder, graph=tf.get_default_graph())
    saver = tf.train.Saver()
    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = W.name
    embedding.metadata_path = os.path.join(model_folder, "metadata.tsv")
    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(writer, config)

    with tf.Session() as sess:
        sess.run(embedding_init, feed_dict={embedding_placeholder: weights})
        save_path = saver.save(sess, os.path.join(model_folder, "tf-model.cpkt"))

    return save_path

    parser.add_argument("--folder", default="models/movie_reviews")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--max_vocab_size", type=int, default=None)
    parser.add_argument("--sample", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min_alpha", type=float, default=0.0001)
    parser.add_argument("--sg", type=int, default=0)
    parser.add_argument("--hs", type=int, default=0)
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--cbow_mean", type=int, default=1)
    parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--null_word", type=int, default=0)


if __name__ == "__main__":
    import json
    import shutil
    from types import SimpleNamespace

    params = SimpleNamespace(folder="testw2v",
                             size=20,
                             alpha=0.025,
                             window=5,
                             min_count=5,
                             max_vocab_size=None,
                             sample=1e-3,
                             seed=1,
                             workers=3,
                             min_alpha=0.0001,
                             sg=0,
                             hs=0,
                             negative=10,
                             cbow_mean=1,
                             iter=5,
                             null_word=0)

    unigram_generator = TxtConnector(filepath="data/SMSSpamCollection.txt")
    sentence_generator = Bigram(unigram_generator)


    os.makedirs(params.folder)
    json.dump(vars(params), open(os.path.join(params.folder, "params.json"), 'w', encoding='utf-8'), indent=2)

    w2v = Word2Vec(save_folder=params.folder)
    w2v.fit(sentence_generator,
            size=params.size,
            alpha=params.alpha,
            window=params.window,
            min_count=params.min_count,
            max_vocab_size=params.max_vocab_size,
            sample=params.sample,
            seed=params.seed,
            workers=params.workers,
            min_alpha=params.min_alpha,
            sg=params.sg,
            hs=params.hs,
            negative=params.negative,
            cbow_mean=params.cbow_mean,
            iter=params.iter,
            null_word=params.null_word)

    create_embeddings(gensim_model=w2v.model, model_folder=params.folder)

    shutil.rmtree(params.folder)
