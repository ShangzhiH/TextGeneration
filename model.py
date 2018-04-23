# encoding=utf-8

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import CoupledInputForgetGateLSTMCell
from layer import Dense, Mask
from attention import attention
import numpy as np
from tensorflow.python.layers import core as layers_core


class Model(object):
    def __init__(self, config, session):
        self.config = config
        self.lr = config["lr"]

        self.hidden_dim = config["hidden_dim"]
        self.dropout = config["dropout"]

        self.char_dim = config["char_dim"]
        self.char_num = config["char_num"]

        self.begin_id = config["begin_symbol_id"]
        self.end_id = config["end_symbol_id"]


        with tf.variable_scope("review_generation_by_attribute"):
            self.merger_train_loss = []

            self.global_step = tf.Variable(0, trainable=False)
            self.global_epoch = tf.Variable(0, trainable=False)

            self.initializer = xavier_initializer()

            # placeholder
            self.sentences = tf.placeholder(dtype = tf.int32, shape = [None, None], name = "Sentences")
            self.last_chars = tf.placeholder(dtype = tf.int32, shape = [None], name = "LastChars")
            self.dropout = tf.placeholder(dtype=tf.float32, name = "Dropout")
            self.batch_size = tf.placeholder(tf.int32, [])
            self.mode = tf.placeholder(tf.string)

            real_char = tf.sign(self.sentences)
            char_len = tf.reduce_sum(real_char, reduction_indices = 1)

            # embedding layer
            self.char_embedding = self.char_embedding_layer(self.sentences, self.dropout)

            # decoder
            # batch_size * max_Q_len * hidden
            helper = tf.contrib.seq2seq.TrainingHelper(self.char_embedding, char_len, time_major=False)

            # batch_size * max_Q_len * V_size
            self.output = self.decoder(helper, self.hidden_dim, name="decoder_for_train")
            # loss
            self.loss = self.loss_layer(self.output, char_len)
            self.merger_train_loss_op = tf.summary.merge(self.merger_train_loss)
            self.opt = tf.train.AdamOptimizer(self.lr)
            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

            # saver of the model
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            initial_state_inference = self.initial_state_for_inference(self.char_embedding, char_len - 1)

            self.output_predict = self.beam_search_decoder(initial_state_inference, self.last_chars, max_iter = 100, name = "decoder_for_inference")


            self.session = session
            self.summary_writer = tf.summary.FileWriter(self.config['summary_path'], self.session.graph)


    def char_embedding_layer(self, char_inputs, dropout = None, name = None):
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(name = "char_embedding", shape = [self.char_num, self.char_dim], initializer = self.initializer)

            embedding = tf.nn.embedding_lookup(self.char_lookup, char_inputs)

            embedding = tf.concat(embedding, axis = 1)

        embedding = tf.nn.dropout(embedding, keep_prob = dropout)
        return embedding

    def lstm_layer(self, lstm_inputs, initial_state, lstm_dim, lengths, name = None):
        with tf.variable_scope("lstm_decoder_layer" if not name else name, reuse = tf.AUTO_REUSE):
            lstm_cell = CoupledInputForgetGateLSTMCell(lstm_dim, use_peepholes = True, initializer = self.initializer, state_is_tuple = True)
            outputs, final_states  = tf.nn.dynamic_rnn(lstm_cell, lstm_inputs, dtype = tf.float32, sequence_length = lengths, initial_state = initial_state)
        return outputs, final_states

    def mult_layer_lstm(self, initial_states, char_embedding, lengths):
        final_states = []
        decoder_output = char_embedding
        for i in range(self.decoder_layer_num - 1):
            decoder_input = tf.nn.dropout(decoder_output, self.dropout)
            with tf.variable_scope("mult_layer_lstm_decoder_%i_layer" % i):
                decoder_output, final_state = self.lstm_layer(decoder_input, initial_states[i], self.hidden_dim, lengths)
                final_states.append(final_state)
        return decoder_output, final_states

    def single_lstm_cell(self, hidden_dim):
        lstm_cell = CoupledInputForgetGateLSTMCell(hidden_dim, use_peepholes=True, initializer = self.initializer, state_is_tuple = True, reuse = False)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = self.dropout)
        return lstm_cell

    def decoder(self, helper, lstm_dim, name = None, max_iter = None):
        with tf.variable_scope("decoder" if not name else name):
            self.cell = self.single_lstm_cell(lstm_dim)
            decoder = tf.contrib.seq2seq.BasicDecoder(self.cell, helper, output_layer = layers_core.Dense(self.char_num, use_bias=False),
                                                      initial_state = self.cell.zero_state(self.batch_size, tf.float32))

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major = False, impute_finished = True, maximum_iterations = max_iter)
        return outputs.rnn_output

    def initial_state_for_inference(self, lstm_inputs, lengths):
        with tf.variable_scope("intialization_for_inference"):
            _, final_states = tf.nn.dynamic_rnn(self.cell, lstm_inputs, dtype = tf.float32, sequence_length = lengths)
        return final_states

    def beam_search_decoder(self, initial_state, start_token, name=None, max_iter=None):
        with tf.variable_scope("beam_search_decoder" if not name else name):
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(initial_state, multiplier = 10)
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(self.cell, embedding=self.char_lookup,
                                                                      start_tokens=start_token,
                                                                      end_token=self.end_id, output_layer = layers_core.Dense(self.char_num, use_bias=False),
                                                      initial_state = decoder_initial_state, beam_width = 10)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major = False, impute_finished = False, maximum_iterations = max_iter)
            result = tf.transpose(outputs.predicted_ids, [0,2,1])
        return result


    def loss_layer(self, logits, seq_length = None):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.sentences, logits = logits)
        if seq_length is not None:
            crossent = tf.expand_dims(crossent, axis = 2)
            crossent = Mask(crossent, seq_length)
        train_loss = tf.reduce_sum(crossent) / (tf.cast(self.batch_size, dtype=tf.float32))

        self.merger_train_loss.append(tf.summary.scalar("train_loss_function", train_loss))

        return train_loss

    def run_step(self, is_train, batch):
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            merged_summary, global_step, loss, _ = self.session.run(
                [self.merger_train_loss_op, self.global_step, self.loss, self.train_op], feed_dict)
            self.summary_writer.add_summary(merged_summary, global_step)
            return global_step, loss
        else:
            logits = self.session.run([self.output_predict], feed_dict)
            return logits

    def create_feed_dict(self, is_train, batch):
        string, chars, last_chars = batch
        feed_dict = {
            self.batch_size: len(batch[0]),
            self.dropout: 1.0,
            self.sentences: chars,
            self.last_chars: last_chars,
            self.mode: "infer",
        }
        if is_train:
            feed_dict[self.mode] = "train"
            feed_dict[self.dropout] = self.config["dropout"]
        return feed_dict

    def save_dev_test_summary(self):
        merged_summary, global_epoch = self.session.run([self.merger_dev_test_evaluation_op, self.global_epoch])
        self.summary_writer.add_summary(merged_summary, global_epoch)

    def evaluate(self, data_manager, id_to_char):
        results = []
        for batch in data_manager.iter_batch():
            scores = self.run_step(False, batch)
            predict_senteces = self.index_to_sentence(scores, id_to_char)
            for sample in range(len(batch[0])):
                sentences = batch[0][sample]

                result = []
                for one_beam_result in predict_senteces[sample]:
                    one_line = ""
                    one_line += u"".join([word for word in sentences if type(word) != int])
                    one_line += u"\t"
                    one_line += u"".join(one_beam_result)
                    result.append(one_line)
                results.append(result)

        return results

    # beam search result to sentences
    # return batch_size * beam_width sentences
    def index_to_sentence(self, scores, id_to_char):
        predict_sentences = []
        for beam_lines in scores[0]:
            predict_sentence = []
            for line in beam_lines:
                sentence = [id_to_char[char_id] for char_id in line if char_id != -1]
                predict_sentence.append(sentence)
            predict_sentences.append(predict_sentence)
        return predict_sentences

