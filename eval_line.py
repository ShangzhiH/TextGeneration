# encoding=utf-8
from utils import load_config, get_logger, load_model
from data_utils import replace_html, full_to_half
import tensorflow as tf
import pickle
import re
from model import EvalModel

if __name__ == "__main__":
    configfile_path = "config/config_file"
    logfile_path = "log/train.log"
    mapfile_path = "maps/maps.pkl"
    config = load_config(configfile_path)
    logger = get_logger(logfile_path)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.gfile.GFile(mapfile_path, "rb") as f:
        char_to_id, id_to_char = pickle.load(f)

    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(char_to_id.keys(), char_to_id.values()),
        char_to_id.get(u"<UNK>")
    )
    x = tf.placeholder(tf.string, shape = [1, None])
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.batch(1).map(lambda token_list: table.lookup(token_list)).map(lambda x: (tf.shape(x)[0], x))

    iter = dataset.make_initializable_iterator()
    eval_model = EvalModel(config, iter)
    with tf.Session(config=tf_config) as sess:
        sess.run(table.init)
        load_model(sess, eval_model, "ckpt/", logger)
        while True:
            sentence = raw_input(u"请输入句子开头, 自由生成则直接回车")
            if type(sentence) == str:
                sentence = sentence.decode("utf-8")

            sentence = sentence.strip()

            if config["zeros"]:
                sentence = re.sub('\d', '0', sentence)
            if config["lower"]:
                sentence = sentence.lower()
                sentence = replace_html(full_to_half(sentence))

            # 空字符处理
            sentence = sentence.replace(u" ", u"#")

            sentence_list = ["<begin>"] + list(sentence) + ["</begin>"]

            sess.run(iter.initializer, feed_dict={x: [sentence_list]})
            print(sess.run(eval_model.perplexity))
