# encoding=utf-8

"""
该文件提供一些load数据, 处理数据集的函数
"""

import sys
import tensorflow as tf
import re
from data_utils import iob2, iob_iobes, create_dico, create_mapping, get_seg_features

reload(sys)
sys.setdefaultencoding("utf-8")

# not implement lowercase here
# because "tag-type check" recognize uppercase tag like ORG etc.
class SentenceGenerator(object):
    def __init__(self, path, zeros):
        self.path = path
        self.zeros = zeros
        self.length = None

    def __len__(self):
        if self.length:
            return self.length
        num = 0
        for line in tf.gfile.GFile(self.path, "r"):
            if line.strip():
                num += 1
        self.length = num
        return self.length

    def __call__(self):
        """
        读入已经是 attributes - sentence 格式的数据

        :param path: 文件路径
        :param lower: 是否转成小写
        :param zeros: 是否用0替换数字
        :return: [ [ [w1-tag], [w2-tag] ...], []]
        """

        num = 0
        for line in tf.gfile.GFile(self.path, "r"):
            if type(line) == str:
                line = line.decode("utf-8")

            if not line.strip():
                continue

            if self.zeros:
                sentence = re.sub('\d', '0', line.strip())
            else:
                sentence = sentence.strip()

            # 空字符
            sentence = sentence.replace(u" ", u"#")

            sentence_list = ["<begin>"] + list(sentence) + ["</begin>"]

            yield sentence_list
        self.length = num


def char_mapping(sentences, lower):
    """
    根据字创建一个字典
    :param sentences: 
    :param lower: 
    :return: 字典，字-id映射，id-字映射
    """

    # 只包含当前词 [[w1, w2,...], [w1, w2, ...], []]
    chars = [[char.lower() if lower else char for char in sentence] for sentence in sentences]
    dico = create_dico(chars)

    dico[u"<PAD>"] = 10000001 # 目前未用到 定义一个大的数保证其对应id为0
    dico[u"<UNK>"] = 10000000 # 未登录词

    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (len(dico), sum(len(x) for x in chars)))

    return dico, char_to_id, id_to_char

class BatchManager(object):

    # data is a generator
    def __init__(self, data_generator, batch_size, lower, char_to_id, train):
        self.data_generator = data_generator() # conserve the generator
        self.batch_size = batch_size
        self.char_to_id = char_to_id
        self.lower = lower
        self.train = train
        self.length = len(data_generator)
        self.len_data = self.length / self.batch_size

    def reset(self, data_generator):
        self.data_generator = data_generator

    # 先根据句子长度排序 再填充
    def sort_and_pad(self, data):
        sorted_data = sorted(data, key=lambda x: len(x[1]))
        return self.pad_data(sorted_data)

    def pad_data(self, data):
        strings = []
        chars = []
        last_chars = []
        max_sentence_length = max([len(sentence[0]) for sentence in data])

        for line in data:
            string, char = line
            last_chars.append(char[-1])
            # 这里采用直接在后面填充0 是否有问题
            padding = [0] * (max_sentence_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
        return [strings, chars, last_chars]

    def prepare_dataset(self, sentences):
        """
        返回
        - word list
        - word indexs list
        - word seg indexs
        - tag indexes
        :param sentences: generator
        :param char_to_id: 
        :param tag_to_id: 
        :param lower: 
        :param train: 
        :return: [[[words], [word_indexs], [word_segs], [tag_indexs]], [[words], [word_indexs], [word_segs], [tag_indexs]], []]
        """
        neg_index = 0

        def f(x):
            return x.lower() if self.lower else x

        data = []
        for sentence in sentences:
            if self.train:
                string = [word_token for word_token in sentence]
                chars = [self.char_to_id[f(word) if f(word) in self.char_to_id else u"<UNK>"] for word in string]
            else:
                string = [word_token for word_token in sentence[1:-1]]
                chars = [self.char_to_id[f(word) if f(word) in self.char_to_id else u"<UNK>"] for word in string]

            data.append([string, chars])
        return data

    def iter_batch(self):
        batch_data = []
        for index, data in enumerate(self.data_generator):
            batch_data.append(data)
            if (index + 1) % self.batch_size == 0:
                yield self.sort_and_pad(self.prepare_dataset(batch_data))
                batch_data = []
        if len(batch_data) > 0:
            yield self.sort_and_pad(self.prepare_dataset(batch_data))





