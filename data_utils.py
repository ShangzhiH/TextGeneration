# encoding=utf-8

"""
该文件提供一些处理数据的函数
"""

import sys
import tensorflow as tf
import numpy as np
import re

reload(sys)

sys.setdefaultencoding("utf-8")


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def create_dico(item_list):
    """
    根据给定的字组成的列表构建字典
    并记录频次
    :param item_list: 
    :return: 
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            dico[item] = dico.get(item, 0) + 1
    return dico


def create_mapping(dico):
    """
    构建字-id和id-字映射
    :param dico: 
    :return: 
    """
    sorted_items = sorted(dico.items(), key = lambda x: (-x[1], x[0]))
    id_to_item = {id: item[0] for id, item in enumerate(sorted_items) if item[1] > 0} # item[0]是字 item[1]是字频
    item_to_id = {item: id for id, item in id_to_item.items()}
    return item_to_id, id_to_item


def get_seg_features(string):
    """
    分词
    :param string: 
    :return: 
    """
    seg_feature = []

    # To Do
    return seg_feature


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    load pretrained embedding vectors
    :param emb_path: 
    :param id_to_word: 
    :param word_dim: 
    :param old_weights: 
    :return: 
    """
    new_weights = old_weights
    print("Loading pretrained embeddings from {}...".format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(tf.gfile.GFile(emb_path, "r")):
        line = line.rstrip().decode("utf-8").split("\t")
        if len(line) == 2 and len(line[1].split(",")) == word_dim:
            pre_trained[line[0]] = np.array([float(x) for x in line[1].split(",")]).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print("Warning: %i invalid lines in embedding file" % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print("Loaded %i pretrained embeddings." % len(pre_trained))
    print("%i / %i (%.4f%%) words have been initialized with pretrained embeddings." % (c_found + c_lower + c_zeros, n_words, 100.0 * (c_found + c_lower + c_zeros) / n_words))
    print('%i found directly, %i after lowercasing, '
          '%i after lowercasing + zero.' % (
              c_found, c_lower, c_zeros
          ))
    return new_weights

def full_to_half(s):
    """
    Convert full-width character to half-width one 
    """
    if type(s) == str:
        s = s.decode("utf-8")
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = unichr(num)
        n.append(char)
    return u''.join(n)


def replace_html(s):
    if type(s) == str:
        s = s.decode("utf-8")

    s = s.replace(u'&quot;',u'"')
    s = s.replace(u'&amp;',u'&')
    s = s.replace(u'&lt;',u'<')
    s = s.replace(u'&gt;',u'>')
    s = s.replace(u'&nbsp;',u' ')
    s = s.replace(u"&ldquo;", u"“")
    s = s.replace(u"&rdquo;", u"”")
    s = s.replace(u"&mdash;",u"")
    s = s.replace(u"\xa0", u" ")
    return(s)


def input_from_line(line, char_to_id):
    """
    take sentence data and return an model input
    :param line: 
    :param char_to_id: 
    :param use_html_tag: 
    :return: 
    """

    if type(line) == str:
        line = line.decode("utf-8")
        line = full_to_half(line)
        line = replace_html(line)
        line = line.replace(u" ", u"#")


    #char_list = list(left) + [u"<dish>"] + list(term) + [u"</dish>"] + list(right)
    char_list = [u"<begin>"] + list(line)
    inputs = [[char_list]]
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id[u"<UNK>"] for char in char_list]])
    inputs.append([inputs[-1][-1][-1]])
    return inputs
