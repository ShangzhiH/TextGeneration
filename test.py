# encoding=utf-8
import io
import json
import tensorflow as tf


if __name__ == "__main__":
    ckpt = tf.train.get_checkpoint_state("ckpt/")
    with tf.Graph().as_default():
        with tf.Graph().as_default():
            with tf.Session() as sess:
                a = tf.placeholder(shape=[10,10], dtype=tf.float32)
                dropout = tf.Variable(0.0, dtype=tf.float32, trainable=False)
                s = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                s.restore(sess, ckpt.model_checkpoint_path)
                print(sess.run(dropout))
