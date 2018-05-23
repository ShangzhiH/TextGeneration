# encoding=utf-8

import os
from collections import OrderedDict
import tensorflow as tf
import pickle
import numpy as np
from utils import make_path, get_logger, load_config, save_config, print_config, create_model, save_model, clean_map, clean, load_model
from dataset_loader import dataset_from_file, line_num_count, MappingInfo
from model import TrainModel, EvalModel, InferModel



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
        self.FLAGS.result_path = os.path.join(FLAGS.root_path, FLAGS.result_path)
        self.FLAGS.train_file = os.path.join(FLAGS.data_root_path, FLAGS.train_file)
        self.FLAGS.dev_file = os.path.join(FLAGS.data_root_path, FLAGS.dev_file)
        self.FLAGS.test_file = os.path.join(FLAGS.data_root_path, FLAGS.test_file)

    def config(self):
        config = OrderedDict()
        config["char_num"] = len(MappingInfo.char_to_id)
        config["char_dim"] = self.FLAGS.char_dim
        config["hidden_dim"] = self.FLAGS.hidden_dim
        config["rnn_layer_num"] = self.FLAGS.rnn_layer_num
        config["infer_num"] = self.FLAGS.infer_num
        config["batch_size"] = self.FLAGS.batch_size

        config["start_symbol_id"] = MappingInfo.char_to_id['<begin>']
        config["end_symbol_id"] = MappingInfo.char_to_id['</begin>']

        config["clip"] = self.FLAGS.clip
        config["use_train_sampling"] = self.FLAGS.use_train_sampling
        config["train_sample_prob"] = self.FLAGS.train_sample_prob
        config["dropout"] = self.FLAGS.dropout
        config["lr"] = self.FLAGS.lr
        config["zeros"] = self.FLAGS.zeros
        config["lower"] = self.FLAGS.lower

        config["summary_path"] = self.FLAGS.summary_path
        return config

    def infer(self, session, model, logger):
        sentence_list = model.infer(session)
        sentence = u"\n".join([u"".join(s) for s in sentence_list])
        logger.info(sentence)

    def evaluate(self, session, model, name, iter_init_op, logger):
        logger.info("evaluate:{}".format(name))
        session.run(iter_init_op) # initilize dev or test iterator
        logger.info("iterator is switched to {}".format(name))

        perplexity = model.evaluate(session)
        logger.info("current {} perplexity score:{:>.3f}".format(name, perplexity))
        if name == "dev":
            self.train_session.run(tf.assign(self.train_model.dev_perplexity, perplexity))
            best_dev_perplexity = self.train_session.run(self.train_model.best_dev_perplexity)
            if perplexity < best_dev_perplexity:
                self.train_session.run(tf.assign(self.train_model.best_dev_perplexity, perplexity))
                logger.info("new best dev perplexity score:{:>.3f}".format(perplexity))
            return (perplexity < best_dev_perplexity, perplexity)
        elif name == "test":
            self.train_session.run(tf.assign(self.train_model.test_perplexity, perplexity))
            best_test_perplexity = self.train_session.run(self.train_model.best_test_perplexity)
            if perplexity < best_test_perplexity:
                self.train_session.run(tf.assign(self.train_model.best_test_perplexity, perplexity))
                logger.info("new best test perplexity score:{:>.3f}".format(perplexity))
            return (perplexity < best_test_perplexity, perplexity)


    def train(self):
        make_path(self.FLAGS)

        logger = get_logger(self.FLAGS.logfile_path)

        # build char-id mapping
        MappingInfo.char_mapping(self.FLAGS.train_file, self.FLAGS.zeros, self.FLAGS.lower)
        MappingInfo.save_map(self.FLAGS.mapfile_path, self.FLAGS.vocabfile_path, logger)

        # load config and print it
        if tf.gfile.Exists(self.FLAGS.configfile_path):
            config = load_config(self.FLAGS.configfile_path)
        else:
            config = self.config()
            save_config(config, self.FLAGS.configfile_path)
        print_config(config, logger)

        # calculate sentence num
        logger.info("Calculating sentence num in dataset")
        train_sentence_num = line_num_count(self.FLAGS.train_file)
        dev_sentence_num = line_num_count(self.FLAGS.dev_file)
        test_sentence_num = line_num_count(self.FLAGS.test_file)
        logger.info("{} / {} / {} sentences in train / dev / test.".format(train_sentence_num, dev_sentence_num,
                                                                           test_sentence_num))

        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            table_train = MappingInfo.make_table_tensor()
            # load data sets
            # use generator to avoid memory oversize
            train_dataset = dataset_from_file(self.FLAGS.train_file, self.FLAGS.zeros, self.FLAGS.lower, self.FLAGS.batch_size, None, table_train)
            logger.info("Train sentence dataset is initialized")
            # build iterator from dataset
            iter_train = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            train_init_op = iter_train.make_initializer(train_dataset)

            self.train_model = TrainModel(config, iter_train)

        self.eval_graph = tf.Graph()
        with self.eval_graph.as_default():
            table_eval = MappingInfo.make_table_tensor()
            dev_dataset = dataset_from_file(self.FLAGS.dev_file, self.FLAGS.zeros, self.FLAGS.lower, self.FLAGS.batch_size, 1, table_eval)
            logger.info("Dev sentence dataset is initialized")
            test_dataset = dataset_from_file(self.FLAGS.test_file, self.FLAGS.zeros, self.FLAGS.lower, self.FLAGS.batch_size, 1, table_eval)
            logger.info("Test sentence dataset is initialized")

            iter_eval = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            dev_init_op = iter_eval.make_initializer(dev_dataset)
            test_init_op = iter_eval.make_initializer(test_dataset)
            eval_model = EvalModel(config, iter_eval)

        self.infer_graph = tf.Graph()
        with self.infer_graph.as_default():
            infer_model = InferModel(config)


        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        #tf_config.log_device_placement = True
        steps_per_epoch = train_sentence_num // config["batch_size"] # how many batches in an epoch


        self.train_session = tf.Session(config = tf_config, graph = self.train_graph)
        eval_session = tf.Session(config = tf_config, graph = self.eval_graph)
        infer_session = tf.Session(config = tf_config, graph = self.infer_graph)

        logger.info("start training")
        create_model(self.train_session, self.train_model, self.FLAGS.ckpt_path, logger)
        self.train_session.run(table_train.init)
        self.train_session.run(train_init_op)

        eval_session.run(table_eval.init)
        loss = []
        lr = config["lr"]
        for i in range(self.FLAGS.max_epoch):
            for j in range(steps_per_epoch):
                step, batch_loss = self.train_model.train(self.train_session)
                loss.append(batch_loss)
                sample_prob = max(0.3, config["train_sample_prob"] - 0.2 * step / steps_per_epoch)  # liner decay sample prob
                self.train_session.run(tf.assign(self.train_model.train_sample_prob, sample_prob))

                if step % self.FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info(
                        "iteration:{} step:{}/{}, NER loss:{:>9.6f}, Training Sample prob is now {:>4.2f}".format(
                            iteration, step % steps_per_epoch,
                            steps_per_epoch, np.mean(loss), sample_prob))
                    loss = []

                if step % self.FLAGS.steps_eval == 0:
                    save_model(self.train_session, self.train_model, self.FLAGS.ckpt_path, logger)
                    load_model(eval_session, eval_model, self.FLAGS.ckpt_path, logger)
                    best, current_perplexity = self.evaluate(eval_session, eval_model, "dev", dev_init_op, logger)
                    if best:
                        save_model(self.train_session, self.train_model, self.FLAGS.best_ckpt_path, logger)
                    self.evaluate(eval_session, eval_model, "test", test_init_op, logger)
                    self.train_model.save_dev_test_summary(self.train_session)

                    load_model(infer_session, infer_model, self.FLAGS.ckpt_path, logger)
                    self.infer(infer_session, infer_model, logger)


            lr = max(0.0001, lr / 1.5)
            self.train_session.run(tf.assign(self.train_model.lr, lr))
            logger.info("Epoch {} is finished, rescale learing rate to {}".format(i, lr))

    def evaluate_line(self):
        config = load_config(self.FLAGS.configfile_path)
        logger = get_logger(self.FLAGS.logfile_path)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.gfile.GFile(self.FLAGS.mapfile_path, "rb") as f:
            char_to_id, id_to_char = pickle.load(f)

        with tf.Session(config = tf_config) as sess:

            x = tf.placeholder(tf.string, shape = [1, N])
            dataset = dataset_from_string
            model = load_model(sess, InferModel, self.FLAGS.ckpt_path, config, logger)


    def run(self):
        if self.FLAGS.train:
            if self.FLAGS.clean:
                clean(self.FLAGS)
            if self.FLAGS.clean_map:
                clean_map(self.FLAGS)
            self.train()
        else:
            self.evaluate_line()
