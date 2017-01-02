# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>

@copyright: Copyright (c) 2016, ioSquare SAS. All rights reserved.
The information contained in this file is confidential and proprietary.
Any reproduction, use or disclosure, in whole or in part, of this
information without the express, prior written consent of ioSquare SAS
is strictly prohibited.

@brief:
"""

import os
import json
import time
import argparse
import logging
import regex as re
from models import CsvConnector, TxtConnector, Word2Vec, create_embeddings


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", default="data/movie_reviews.csv")

    parser.add_argument("--input_type", choices=['csv', 'txt'], default='csv', help="The kind of input data")

    parser.add_argument("--separator", type=str, default=',',
                        help="csv separator (only if input_type == 'csv'")

    parser.add_argument("--columns_to_select", nargs='*', default=["Phrase"],
                        help="Columns of your csv to select (only if input_type == 'csv')."
                             "You must select at least one column."),

    parser.add_argument("--columns_joining_token", type=str, default='. ',
                        help="token that will join multiple csv columns (only if input_type == 'csv').")

    parser.add_argument("--folder", default="123")
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
    params = parser.parse_args()
    return params


def preprocessing(sentence):
    """
    Add a custom pre-processing on sentence before feeding them to word2vec
    """
    sentence = sentence.lower()
    sentence = re.sub(r'[^\P{P}\']+', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = "".join(sentence).strip()
    return sentence


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    prefix = int(time.time())

    params = get_args()
    params.folder = os.path.realpath(params.folder)
    print(params)

    if params.input_type == 'csv':
        sentence_generator = CsvConnector(filepath=params.file,
                                          preprocessing=preprocessing,
                                          separator=params.separator,
                                          columns_to_select=params.columns_to_select,
                                          columns_joining_token=params.columns_joining_token)
    else:
        sentence_generator = TxtConnector(filepath=params.file, preprocessing=preprocessing)

    os.mkdir(params.folder)
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

