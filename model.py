# encoding=utf-8

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import LSTMCell
from layer import Dense, Mask
from attention import attention
import numpy as np
from tensorflow.python.layers import core as layers_core
from dataset_loader import MappingInfo


class Model(object):
    def __init__(self, config, mode, iterator = None):
        self.config = config

        self.hidden_dim = config["hidden_dim"]
        self.rnn_layer_num = config["rnn_layer_num"]

        self.char_dim = config["char_dim"]
        self.char_num = config["char_num"]

        self.start_id = config["start_symbol_id"]
        self.end_id = config["end_symbol_id"]

        self.mode = mode

        self.initializer = xavier_initializer()
        if self.mode == "train":
            with tf.variable_scope("train_variable"):
                self.merger_train_loss = []
                self.merger_dev_test_evaluation = []

                self.best_dev_perplexity = tf.Variable(float("inf"), trainable=False)
                self.dev_perplexity = tf.Variable(0.0, trainable=False)
                self.merger_dev_test_evaluation.append(tf.summary.scalar("dev_perplexity", self.dev_perplexity))

                self.best_test_perplexity = tf.Variable(float("inf"), trainable=False)
                self.test_perplexity = tf.Variable(0.0, trainable=False)
                self.merger_dev_test_evaluation.append(tf.summary.scalar("test_perplexity", self.test_perplexity))

                self.merger_dev_test_evaluation_op = tf.summary.merge(self.merger_dev_test_evaluation)

                # no trainable variable
                self.global_step = tf.Variable(0, trainable = False)
                self.lr = tf.Variable(config["lr"], dtype = tf.float32, trainable = False)
                self.use_train_sampling = config["use_train_sampling"]
                self.train_sample_prob = tf.Variable(config["train_sample_prob"], dtype = tf.float32, trainable = False)
                self.dropout = tf.Variable(config["dropout"], dtype = tf.float32, trainable = False)

            self._build_graph(iterator)

        elif self.mode == "eval":
            self.dropout = 1.0
            self._build_graph(iterator)

        elif self.mode == "infer":
            self.dropout = 1.0
            self.infer_num = config["infer_num"]
            self._build_infer_graph()

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def _build_graph(self, iterator):
        self.iterator = iterator
        self.batch_size, self.sentences = self.iterator.get_next()

        real_char = tf.sign(self.sentences)
        char_len = tf.reduce_sum(real_char, reduction_indices = 1)

        # embedding layer
        self.char_embedding = self._build_char_embedding_layer(self.sentences, self.dropout)

        ## train decoder
        # helper
        if self.mode == "train":
            train_helper = self._build_train_helper(char_len)
            # decoder
            self.train_output = self._build_decoder(train_helper, self.batch_size, None, name="decoder").rnn_output
            ## loss
            self.loss = self._build_loss_layer(self.train_output, char_len)
            self.merger_train_loss_op = tf.summary.merge(self.merger_train_loss)
            self.opt = tf.train.AdamOptimizer(self.lr)
            ## train op
            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)
            self.summary_writer = tf.summary.FileWriter(self.config['summary_path'])

        elif self.mode == "eval":
            test_helper = self._build_test_helper(char_len)
            self.test_output = self._build_decoder(test_helper, self.batch_size, None, name = "decoder").rnn_output
            self.perplexity = self._calculate_perplexity(self.test_output, char_len)

    def _build_infer_graph(self):
        with tf.variable_scope("char_embedding"):
            char_lookup = tf.get_variable(name = "char_embedding", shape = [self.char_num, self.char_dim], initializer = self.initializer)
        helper_inference = tf.contrib.seq2seq.SampleEmbeddingHelper(char_lookup,
                                                                        start_tokens=tf.tile([self.start_id],
                                                                                             [self.infer_num]),
                                                                        end_token=self.end_id, softmax_temperature=1.0)
        # self.output_predict = self.beam_search_decoder(initial_state_inference, self.last_chars, max_iter = 100, name = "decoder_for_inference")
        self.output_predict = self._build_decoder(helper_inference, self.infer_num, None, "decoder", 100).sample_id

    def _build_char_embedding_layer(self, char_inputs, dropout = None, name = None):
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(name = "char_embedding", shape = [self.char_num, self.char_dim], initializer = self.initializer)
            embedding = tf.nn.embedding_lookup(self.char_lookup, char_inputs)

            embedding = tf.concat(embedding, axis = 1)

        embedding = tf.nn.dropout(embedding, keep_prob = dropout)
        return embedding

    def _single_lstm_cell(self, hidden_dim):
        lstm_cell = LSTMCell(hidden_dim, use_peepholes=False, initializer = self.initializer, state_is_tuple = True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = self.dropout)
        return lstm_cell

    def _stack_lstm_cell(self):
        cells = [self._single_lstm_cell(self.hidden_dim) for _ in range(self.rnn_layer_num)]
        return tf.nn.rnn_cell.MultiRNNCell(cells, True)

    def _build_train_helper(self, char_len):
        if self.use_train_sampling:
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(self.char_embedding, char_len, self.char_lookup,
                                                                         sampling_probability = 1.0 - self.train_sample_prob,
                                                                     name="ScheduledSampleTrainingHelper")
        else:
            helper = tf.contrib.seq2seq.TrainingHelper(self.char_embedding, char_len, time_major=False, name="TrainingHelper")
        return helper

    def _build_test_helper(self, char_len):
        return tf.contrib.seq2seq.TrainingHelper(self.char_embedding, char_len, time_major = False)

    def _build_decoder(self, helper, batch_size, initial_state = None, name = None, max_iter = None):
        with tf.variable_scope("decoder" if not name else name, reuse=tf.AUTO_REUSE):
            self.cell = self._stack_lstm_cell()
            self.projection = layers_core.Dense(self.char_num, use_bias=False)

            decoder = tf.contrib.seq2seq.BasicDecoder(self.cell, helper, output_layer = self.projection,
                                                      initial_state = self.cell.zero_state(batch_size, tf.float32) if not initial_state else initial_state)

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major = False, impute_finished = True, maximum_iterations = max_iter)
        return outputs

    #def initial_state_for_inference(self, lstm_inputs, lengths):
    #    with tf.variable_scope("intialization_for_inference"):
    #        _, final_states = tf.nn.dynamic_rnn(self.cell, lstm_inputs, dtype = tf.float32, sequence_length = lengths)
    #    return final_states

    #def beam_search_decoder(self, initial_state, start_token, name=None, max_iter=None):
    #    with tf.variable_scope("beam_search_decoder" if not name else name):
    #        decoder_initial_state = tf.contrib.seq2seq.tile_batch(initial_state, multiplier = self.beam_search_width)
    #        decoder = tf.contrib.seq2seq.BeamSearchDecoder(self.cell, embedding=self.char_lookup,
    #                                                                  start_tokens=start_token,
    #                                                                  end_token=self.end_id, output_layer = layers_core.Dense(self.char_num, use_bias=False),
    #                                                  initial_state = decoder_initial_state, beam_width = self.beam_search_width)
    #        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major = False, impute_finished = False, maximum_iterations = max_iter)
    #        result = tf.transpose(outputs.predicted_ids, [0,2,1])
    #    return result


    def _build_loss_layer(self, logits, seq_length = None):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.sentences[:, 1:], logits = logits[:, :-1, :])
        self.crossent = crossent
        if seq_length is not None:
            crossent = tf.expand_dims(crossent, axis = 2)
            crossent = Mask(crossent, seq_length - 1)
        train_loss = tf.reduce_sum(crossent) / (tf.cast(self.batch_size, dtype=tf.float32))

        self.merger_train_loss.append(tf.summary.scalar("train_loss_function", train_loss))

        return train_loss

    def _calculate_perplexity(self, logits, seq_length = None):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.sentences[:, 1:], logits = logits[:, :-1, :])
        if seq_length is not None:
            crossent = tf.expand_dims(crossent, axis = 2)
            crossent = Mask(crossent, seq_length - 1)
        crossent = tf.reduce_sum(crossent, axis = 1)

        if seq_length is not None:
            crossent = tf.transpose(crossent) * 1.0 / tf.cast(seq_length - 1, dtype = tf.float32)
        else:
            crossent = tf.transpose(crossent) * 1.0 / tf.cast(tf.shape(crossent)[1], dtype = tf.float32)

        perplexity = tf.reduce_sum(tf.exp(crossent)) / (tf.cast(self.batch_size, dtype=tf.float32))

        return perplexity

    def train(self, session):
        assert self.mode == "train"
        merged_summary, global_step, loss, _ = session.run(
            [self.merger_train_loss_op, self.global_step, self.loss, self.train_op]
        )
        self.summary_writer.add_summary(merged_summary, global_step)
        return global_step, loss

    def save_dev_test_summary(self, session):
        assert self.mode == "train"
        merged_summary, global_step = session.run(
            [self.merger_dev_test_evaluation_op, self.global_step]
        )
        self.summary_writer.add_graph(session.graph)
        self.summary_writer.add_summary(merged_summary, global_step)



    def evaluate(self, session):
        perplexity_record = []
        try:
            while True:
                perplexity_record.append(session.run(self.perplexity))
        except tf.errors.OutOfRangeError:
            return np.mean(perplexity_record)

    def infer(self, session):
        predict_logit = session.run([self.output_predict])
        sentences = self._logit_to_sentence(predict_logit, MappingInfo.id_to_char)
        return sentences


    #def evaluate_line(self, line_input, id_to_char):
    #    scores = self.run_step(False, line_input)
    #    predict_senteces = self._logit_to_sentence(scores, id_to_char)
    #    return predict_senteces[0][0]

    # beam search result to sentences
    # return batch_size * beam_width sentences
    def _index_to_sentence(self, scores, id_to_char):
        predict_sentences = []
        for beam_lines in scores[0]:
            predict_sentence = []
            for line in beam_lines:
                sentence = [id_to_char[char_id] for char_id in line if char_id != -1]
                predict_sentence.append(sentence)
            predict_sentences.append(predict_sentence)
        return predict_sentences

    # search which returns logits instead of char-id
    def _logit_to_sentence(self, scores, id_to_char):
        predict_sentences = []
        for line in scores[0]:
            predict_sentence = [id_to_char[vec] for vec in line]
            predict_sentences.append(predict_sentence)
        return predict_sentences

class TrainModel(Model):
    def __init__(self, config, iterator):
        super(TrainModel, self).__init__(config, "train", iterator)

class EvalModel(Model):
    def __init__(self, config, iterator):
        super(EvalModel, self).__init__(config, "eval", iterator)

class InferModel(Model):
    def __init__(self, config):
        super(InferModel, self).__init__(config, "infer")















