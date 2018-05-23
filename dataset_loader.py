# encoding=utf-8

"""
该文件利用tf的dataset api进行数据读取
"""

import tensorflow as tf
import sys
import re
from data_utils import full_to_half, replace_html
import numpy as np
import pickle

reload(sys)
sys.setdefaultencoding("utf-8")

class MappingInfo():
    char_to_id = {}
    char_to_id[u"<PAD>"] = 0
    char_to_id[u"<UNK>"] = 1
    char_to_id[u"<begin>"] = 2
    char_to_id[u"</begin>"] = 3

    id_to_char = {}
    id_to_char[0] = u"<PAD>"
    id_to_char[1] = u"<UNK>"
    id_to_char[2] = u"<begin>"
    id_to_char[3] = u"</begin>"

    table = None

    @classmethod
    def char_mapping(cls, file_path, zeros, lower):
        char_freq = {}
        for token_list in _generator_reader(file_path, zeros, lower)():
            for token in token_list:
                char_freq[token] = char_freq.get(token, 0) + 1
            if len(char_freq) >= 15000:
                break
        for key, value in char_freq.items():
            if key not in cls.char_to_id and value >= 0:
                cls.char_to_id[key], cls.id_to_char[len(cls.id_to_char)] = len(cls.char_to_id), key

    @classmethod
    def save_map(cls, mapfile_path, vocabfile_path, logger):
        with tf.gfile.GFile(mapfile_path, "wb") as f:
            pickle.dump([MappingInfo.char_to_id, MappingInfo.id_to_char], f)
        logger.info("Created dictionary of word from train data")
        with tf.gfile.GFile(vocabfile_path, "w") as file:
            for word in MappingInfo.char_to_id:
                file.write(word + "\n")
        logger.info("Created readable vocabulary of word from train data")

    @classmethod
    def make_table_tensor(cls):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(cls.char_to_id.keys(), cls.char_to_id.values()),
            cls.char_to_id.get(u"<UNK>")
        )


def _generator_reader(file_path, zeros, lower):
    def gen():
        for line in tf.gfile.GFile(file_path, "r"):
            if type(line) == str:
                line = line.decode("utf-8")

            sentence = line.strip()
            if not sentence:
                continue

            if zeros:
                sentence = re.sub('\d', '0', sentence)
            if lower:
                sentence = sentence.lower()
                sentence = replace_html(full_to_half(sentence))

            # 空字符处理
            sentence = sentence.replace(u" ", u"#")

            sentence_list = ["<begin>"] + list(sentence) + ["</begin>"]
            yield sentence_list
    return gen

def line_num_count(file_path):
    num = 0
    for line in tf.gfile.GFile(file_path, "r"):
        if line.strip():
            num += 1
    return num

def dataset_from_file(file_path, zeros, lower, batch_size, max_epoch, table):


    dataset = tf.data.Dataset.from_generator(_generator_reader(file_path, zeros, lower), tf.string, tf.TensorShape([None]))
    dataset = dataset.shuffle(buffer_size = 1000)
    dataset = dataset.map(lambda token_list: table.lookup(token_list))

    if max_epoch:
        dataset = dataset.padded_batch(batch_size, padded_shapes = tf.TensorShape([None]))\
            .map(lambda x: (tf.shape(x)[0], x)).repeat(max_epoch)
    else:
        dataset = dataset.padded_batch(batch_size, padded_shapes=tf.TensorShape([None]))\
            .map(lambda x: (tf.shape(x)[0], x)).repeat()

    return dataset




