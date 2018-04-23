# encoding

import tensorflow as tf
from layer import Mask, Dense

# Q batch_size * max_Q_len * d_q
# K batch_size * max_V_len * d_k
# V batch_size * max_V_len * d_v
def attention(Q, K, V, Q_len = None, V_len = None, name = None):
    with tf.variable_scope("decoder_attention" if not name else name):
        batch_size, Q_max_len = tf.shape(Q)[0], tf.shape(Q)[1]
        V_max_len = tf.shape(K)[1]
        d_q = Q.shape[-1].value
        d_k = K.shape[-1].value

        Q = Mask(Q, Q_len)
        # (batch_size, max_Q_len, 1, h) + (batch_size, max_Q_len, V_max_len, d_q)
        Q = tf.expand_dims(Q, 2) + tf.zeros((batch_size, Q_max_len, V_max_len, d_q))

        K = Mask(K, V_len)
        # (batch_size, 1, max_K_len, d_k) + (batch_size, max_Q_len, max_K_len, d_k)
        K = tf.expand_dims(K, 1) + tf.zeros((batch_size, Q_max_len, V_max_len, d_k))

        V = Mask(V, V_len)

        combined = tf.concat([Q, K], axis=3) # batch_size * Q_max_len * V_max_len * (d_q + d_k)

        attention = Dense(combined, d_q + d_k, 1, False, None, "attention_layer") # batch_size * Q_max_len * V_max_len * 1
        attention = tf.tanh(tf.squeeze(attention, axis = 3)) # batch_size * Q_max_len * V_max_len

        softmax = tf.nn.softmax(attention, dim = -1) # batch_size * Q_max_len * V_max_len

        # batch_size * max_Q_len * d_v
        c = tf.matmul(softmax, V)

        return c










