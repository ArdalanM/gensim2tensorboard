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

        if len(columns_to_select) == 0:
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

        if self.phrases:
            # sentences = gensim.models.phrases.Phraser(sentences)
            sentences = gensim.models.Phrases(sentences)

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
                                            null_word=null_word, )

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