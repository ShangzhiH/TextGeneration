# encoding=utf-8

import tensorflow as tf


# inputs 输入 (batch_size, seq_len, input_size) 序列进行了padding
# seq_len 序列的真实长度 (batch_size, )
# 作用是根据mode对输入序列进行一定的处理
# mode = 'mul' 对多余部分置零 用于全链接层之前
# model = 'add' 对多余部分减掉一个很大的数 用于softmax之前 保证被mask的部分近似为 -inf
def Mask(inputs, seq_len, mode='mul'):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), tf.float32) # 对(batch_size,) 的句子长度生成对应的mask (batch_size, max_seq_len)
        for _ in range(len(inputs.shape) - 2):
            mask = tf.expand_dims(mask, 2) # (batch_size, max_seq_len, 1, 1, ...)扩展维度根据inputs维度决定
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12



# position-wise的全联接 transform input_size to output_size
# inputs (batch_size, seq_len, input_size)
def Dense(inputs, input_size, output_size, bias=True, seq_len=None, name = None):
    with tf.variable_scope("projection" if not name else name):
        W = tf.get_variable("W", initializer = tf.random_uniform([input_size, output_size], -0.05, 0.05))
        if bias:
            b = tf.get_variable("b", initializer = tf.random_uniform([output_size], -0.05, 0.05))
        else:
            b = 0

        outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
        outputs = tf.reshape(outputs, tf.concat([tf.shape(inputs)[:-1], [output_size]], 0))

        if seq_len != None:
            outputs = Mask(outputs, seq_len, 'mul')
    return outputs
