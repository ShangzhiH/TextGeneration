# encoding=utf-8

import os
from collections import OrderedDict
import tensorflow as tf
import pickle
import numpy as np

from utils import make_path, get_logger, load_config, save_config, print_config, create_model, save_model, clean_map, clean, test_generation
from data_loader import SentenceGenerator, char_mapping, BatchManager
from model import Model



class ModelUsage(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.FLAGS.ckpt_path = os.path.join(FLAGS.root_path, FLAGS.ckpt_path)
        self.FLAGS.summary_path = os.path.join(FLAGS.root_path, FLAGS.summary_path)
        self.FLAGS.log_path = os.path.join(FLAGS.root_path, FLAGS.log_path)
        self.FLAGS.logfile_path = os.path.join(self.FLAGS.log_path, "train.log")
        self.FLAGS.map_path = os.path.join(FLAGS.root_path, FLAGS.map_path)
        self.FLAGS.mapfile_path = os.path.join(self.FLAGS.map_path, "maps.pkl")
        self.FLAGS.vocab_path = os.path.join(FLAGS.root_path, FLAGS.vocab_path)
        self.FLAGS.vocabfile_path = os.path.join(FLAGS.vocab_path, "vocabulary.csv")
        self.FLAGS.config_path = os.path.join(FLAGS.root_path, FLAGS.config_path)
        self.FLAGS.configfile_path = os.path.join(self.FLAGS.config_path, "config_file")
        self.FLAGS.script_path = os.path.join(FLAGS.root_path, FLAGS.script_path)
        self.FLAGS.result_path = os.path.join(FLAGS.root_path, FLAGS.result_path)
        self.FLAGS.train_file = os.path.join(FLAGS.data_root_path, FLAGS.train_file)
        self.FLAGS.dev_file = os.path.join(FLAGS.data_root_path, FLAGS.dev_file)

    def config(self, char_to_id):
        config = OrderedDict()
        config["char_num"] = len(char_to_id)
        config["char_dim"] = self.FLAGS.char_dim
        config["hidden_dim"] = self.FLAGS.hidden_dim
        config["batch_size"] = self.FLAGS.batch_size

        config["begin_symbol_id"] = char_to_id['<begin>']
        config["end_symbol_id"] = char_to_id['</begin>']

        config["clip"] = self.FLAGS.clip

        config["dropout"] = self.FLAGS.dropout
        config["lr"] = self.FLAGS.lr
        config["zeros"] = self.FLAGS.zeros
        config["lower"] = self.FLAGS.lower

        config["summary_path"] = self.FLAGS.summary_path
        return config

    def evaluate(self, model, name, data, id_to_char, logger):
        logger.info("evaluate:{}".format(name))
        generation_results = model.evaluate(data, id_to_char)
        test_generation(generation_results, self.FLAGS.result_path, logger)

    def train(self):
        make_path(self.FLAGS)

        logger = get_logger(self.FLAGS.logfile_path)

        # load data sets
        # use generator to avoid memory oversize
        train_sentences = SentenceGenerator(self.FLAGS.train_file, self.FLAGS.zeros)
        logger.info("Train sentence generator is initialized")
        dev_sentences = SentenceGenerator(self.FLAGS.dev_file, self.FLAGS.zeros)
        logger.info("Dev sentence generator is initialized")

        # create maps if not exist
        if not tf.gfile.Exists(self.FLAGS.mapfile_path):
            # create dictionary for word
            _, char_to_id, id_to_char = char_mapping(train_sentences(), self.FLAGS.lower)
            logger.info("Created dictionary of word from train data")
        else:
            with tf.gfile.GFile(self.FLAGS.mapfile_path, "rb") as f:
                char_to_id, id_to_char = pickle.load(f)
                logger.info("Load dictionary from existed map file")

        if not tf.gfile.Exists(self.FLAGS.vocabfile_path):
            with tf.gfile.GFile(self.FLAGS.vocabfile_path, "w") as file:
                for word in char_to_id:
                    file.write(word + "\n")
            logger.info("Created vocabulary file")

        # load config and print it
        if tf.gfile.Exists(self.FLAGS.configfile_path):
            config = load_config(self.FLAGS.configfile_path)
        else:
            config = self.config(char_to_id)
            save_config(config, self.FLAGS.configfile_path)
        print_config(config, logger)

        # prepare data
        # get char_based, char_index_based, segs_based, tag_index_based sentences
        # use generator to avoid memory oversize
        train_manager = BatchManager(train_sentences, config['batch_size'], config['lower'], char_to_id, True)
        logger.info("Train manager is initialized")
        dev_manager = BatchManager(dev_sentences, 100, config['lower'], char_to_id, False)
        logger.info("Dev manager is initialized")
        logger.info("{} / {} sentences in train /dev.".format(len(train_sentences), len(dev_sentences)))

        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        # tf_config.log_device_placement = True
        steps_per_epoch = train_manager.len_data  # how many batches in an epoch
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, self.FLAGS.ckpt_path, config, logger)
            logger.info("start training")
            loss = []
            for i in range(self.FLAGS.max_epoch):
                tf.assign(model.global_epoch, i).eval()
                for batch in train_manager.iter_batch():
                    step, batch_loss = model.run_step(True, batch)
                    loss.append(batch_loss)
                    if step % self.FLAGS.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info(
                            "iteration:{} step:{}/{}, NER loss:{:>9.6f}".format(iteration, step % steps_per_epoch,
                                                                                steps_per_epoch, np.mean(loss)))
                        loss = []
                    if step % 10 == 0:
                        self.evaluate(model, "dev", dev_manager, id_to_char, logger)
                        dev_manager.reset(dev_sentences())
                        logger.info("Epoch {} is finished, reset dev_manager".format(i))
                save_model(sess, model, self.FLAGS.ckpt_path + "/s", logger)

                # reset BatchManager
                train_manager.reset(train_sentences())
                logger.info("Epoch {} is finished, reset train_manager".format(i))


    def run(self):
        if self.FLAGS.train:
            if self.FLAGS.clean:
                clean(self.FLAGS)
            if self.FLAGS.clean_map:
                clean_map(self.FLAGS)
            self.train()
        else:
            self.evaluate_line()
