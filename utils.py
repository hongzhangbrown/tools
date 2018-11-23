from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import os
import pickle
import sqlite3
from functools import cmp_to_key
from collections import OrderedDict
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
import subprocess

class EmbeddingTransformer(BaseEstimator,TransformerMixin):
    PAD = '<PAD>'
    UNK = '<unk>'
    PREFIX = '<s>'
    SUFFIX = '</s>'
    logger = logging.getLogger('EmbeddingLogger')

    def __init__(self, filename, limit=None, dtype=np.float16, enable_cache=True):
        self.dtype = dtype
        self.filename = filename
        self.limit = limit
        self.enable_cache = enable_cache
        self.dictionary = {}
        self.embeddings = np.zeros((0,0),dtype=dtype)
        self.loaded = False
        self.load()
        self.inverse_dictionary = {offset: word for word,offset in self.dictionary.items()}

    @property
    def padding(self):
        return self.dictionary[self.PAD]
    @property
    def oov(self):
        return self.dictionary[self.UNK]
    @property
    def prefix(self):
        return self.dictionary[self.PREFIX]

    @property
    def suffix(self):
        return self.dictionary[self.SUFFIX]

    @staticmethod
    def save_pickle(data,filename):
        with tf.gfile.Gfile(filename,'wb') as pickleFile:
            pickle.dump(data, pickleFile, protocol=2)

    @staticmethod
    def load_pickle(filename):
        with tf.gfile.Gfile(filename,'rb') as pickleFile:
            return pickle.load(pickleFile)

    def init_embedding(self,length,width):
        builtin_tokens = [self.PAD, self.UNK, self.PREFIX, self.SUFFIX]
        for token in builtin_tokens:
            self.dictionary[token] = len(self.dictionary)
        dict_length = length if not self.limit or length <= self.limit else self.limit
        self.embeddings = np.zeros([dict_length + len(builtin_tokens), width], dtype=self.dtype)
        self.embeddings[self.dictionary[self.PAD]].fill(0)
        self.embeddings[self.dictionary[self.UNK]].fill(1e-5)
        self.embeddings[self.dictionary[self.PREFIX]].fill(1e-5)
        self.embeddings[self.dictionary[self.SUFFIX]].fill(1e-5)
        return dict_length


    def load_from_cache(self):
        cache_dir = '{}.cache'.format(self.filename)
        prefix = str(self.limit) if self.limit else 'all'
        meta_file_name = '{}.meta'.format(prefix)
        mmap_file_name = '{},mmap'.format(prefix)
        meta_file_path = os.path.join(cache_dir, meta_file_name)
        mmap_file_path = os.path.join(cache_dir, mmap_file_name)

        #create cache directory
        required_files = [cache_dir, meta_file_path, mmap_file_path]
        for file_name in required_files:
            if not os.path.exists(file_name):
                return False

        logging.debug("load emeddings from cache")

        meta = self.load_pickle(meta_file_path)
        self.dictioanry = meta['dictionary']
        shape = meta['shape']

        #load mmap
        mmap_fp = np.memmap(mmap_file_path, dtype=self.dtype, mode='r', shape=shape)
        self.embeddings = mmap_fp
        return True


    def save_to_cache(self):

        temp_path = './tmp/mmap'
        cache_dir = '{}.cache'.format(self.filename)
        prefix = str(self.limit) if self.limit else 'all'

        meta_file_name = '{}.meta'.format(prefix)
        mmap_file_name = '{},mmap'.format(prefix)
        meta_file_path = os.path.join(cache_dir, meta_file_name)
        mmap_file_path = os.path.join(cache_dir, mmap_file_name)

        #create cache directory
        if not tf.gfile.Exists(cache_dir):
            tf.gfile.MKdir(cache_dir)

        logging.debug("save embedding to cache")
        #save meta

        self.save_pickle({'dictionary':self.dictionary, 'shape': self.embeddings.shape}, meta_file_path)

        #save mmap

        mmap_fp = np.memmap(temp_path, dtype=self.dtype, mode='w+', shape=self.embeddings.shape)
        mmap_fp[:] = self.embeddings[:]
        mmap_fp.flush()
        del mmap_fp
        if not tf.gfile.Exists(mmap_file_path):
            subprocess.call(['hadoop','fs','-copyFromLocal',temp_path, mmap_file_path])



    def load(self):
        if self.enable_cache and self.load_from_cache():
            return
        logging.info('start loading embedding')
        (length, width) = self.get_embedding_shape(self.filename)
        actual_length = self.init_embedding(length,width)
        logging.debug('embedding vocabulary limit:{}'.format(actual_length))
        self.load_embeddings(actual_length)
        self.loaded = True
        logging.info('finished loading embedding')
        if self.enable_cache:
            self.save_to_cache()

    @staticmethod
    def get_embedding_shape(filename):
        length = 0
        width = 0
        with tf.gfile.Gfile(filename,'rb') as file:
            for line in file:
                length += 1
                if not width:
                    (_, values) = line.decode('utf-8').strip().split(" ",1)
                    width = np.fromstring(values, sep=' ').shape[0]
        return length,width

    def load_embeddings(self, limit=None):
        count = 0
        with tf.gfile.Gfile(self.filename,'rb') as file:
            for line in file:
                (word,values) = line.decode('utf-8').strip().split(' ', 1)
                offset = len(self.dictionary)
                self.dictionary[word] = offset
                self.embeddings[offset] = np.fromstring(values, dtype= self.dtype, sep=' ')
                count += 1
                if limit and count >= limit:
                    break

    def fit(self,y):
        if not self.loaded:
            self.load()
        return self

    def transform(self,y):
        unk_vector = self.dictionary[self.UNK]
        data = []
        for sentence in y:
            data.append([self.dictioanry.get(token, unk_vector) for token in sentence])
        return data

    def inverse_transform(self,y):
        return [self.inverse_dictionary[index] for index in y]


def dropout_tf(dropout_rate, mode):
    def positive():
        return dropout_rate
    def negative():
        return 0.0
    dropout = tf.cond(mode,positive,negative)
    return dropout
