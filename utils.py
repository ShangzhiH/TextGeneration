# encoding=utf-8

"""
该文件提供一些与数据无关的工具类函数
"""

import sys
import os
import shutil
import tensorflow as tf
import json
import logging
from bleu import compute_bleu



reload(sys)
sys.setdefaultencoding("utf-8")



def make_path(params):
    """
    Create folder if not exists
    :param params: 
    :return: 
    """
    if not tf.gfile.IsDirectory(params.ckpt_path):
        tf.gfile.MakeDirs(params.ckpt_path)
    if not tf.gfile.IsDirectory(params.summary_path):
        tf.gfile.MakeDirs(params.summary_path)
    if not os.path.isdir(params.log_path):
        os.makedirs(params.log_path)
    if not tf.gfile.IsDirectory(params.map_path):
        tf.gfile.MakeDirs(params.map_path)
    if not tf.gfile.IsDirectory(params.vocab_path):
        tf.gfile.MakeDirs(params.vocab_path)
    if not tf.gfile.IsDirectory(params.script_path):
        tf.gfile.MakeDirs(params.script_path)
    if not tf.gfile.IsDirectory(params.result_path):
        tf.gfile.MakeDirs(params.result_path)
    if not tf.gfile.IsDirectory(params.config_path):
        tf.gfile.MakeDirs(params.config_path)


def clean_map(params):
    """
    keep last time map can save much time
    :param params: 
    :return: 
    """
    if tf.gfile.IsDirectory(params.vocab_path):
        tf.gfile.DeleteRecursively(params.vocab_path)

    if tf.gfile.IsDirectory(params.map_path):
        tf.gfile.DeleteRecursively(params.map_path)

def clean(params):
    """
    Clean current folder
    remove saved model and training log
    :param params: 
    :return: None 
    """
    if tf.gfile.IsDirectory(params.ckpt_path):
        tf.gfile.DeleteRecursively(params.ckpt_path)

    if tf.gfile.IsDirectory(params.summary_path):
        tf.gfile.DeleteRecursively(params.summary_path)

    if tf.gfile.IsDirectory(params.result_path):
        tf.gfile.DeleteRecursively(params.result_path)

    if tf.gfile.IsDirectory(params.log_path):
        tf.gfile.DeleteRecursively(params.log_path)

    if tf.gfile.IsDirectory("__pycache__"):
        tf.gfile.DeleteRecursively("__pycache__")

    if tf.gfile.IsDirectory(params.config_path):
        tf.gfile.DeleteRecursively(params.config_path)


def save_config(config, config_file):
    with tf.gfile.GFile(config_file, "w") as f:
        f.write(unicode(json.dumps(config, ensure_ascii=False, indent=4)))


def load_config(config_file):
    with tf.gfile.GFile(config_file, "r") as f:
        return json.load(f)


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def print_config(config, logger):
    """
    Print configuration of the model
    """
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))

def create_model(session, Model_class, path, config, logger):
    # create model, reuse parameters if exists
    model = Model_class(config, session)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def save_model(sess, model, path, logger):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")


def export_model(sess, model, path, version, logger, id_to_tag, char_to_id):
    export_path = os.path.join(path, str(version))
    if tf.gfile.IsDirectory(export_path):
        tf.gfile.DeleteRecursively(export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(model.char_inputs)
    tensor_info_logits = tf.saved_model.utils.build_tensor_info(model.logits)
    tensor_info_trans = tf.saved_model.utils.build_tensor_info(model.trans)
    tensor_info_dropout = tf.saved_model.utils.build_tensor_info(model.dropout)

    predict_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs = {'sentences': tensor_info_x, 'dropout': tensor_info_dropout},
            outputs = {'logits': tensor_info_logits, 'trans': tensor_info_trans},
            method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    )

    legacy_init_op = tf.group(tf.tables_initializer(), name = 'legacy_init_op')

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map = {
            'predict_tags':
                predict_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            predict_signature,
        },
        legacy_init_op = legacy_init_op
    )
    builder.save()
    logger.info('Done exporting')

    with tf.gfile.GFile(os.path.join(export_path, "char_to_id.csv"), "w") as file:
        for key, value in char_to_id.iteritems():
            file.write("%s\t%s\n" % (key, value))
    with tf.gfile.GFile(os.path.join(export_path, "id_to_tag.csv"), "w") as file:
        for key, value in id_to_tag.iteritems():
            file.write("%s\t%s\n" % (key, value))


def result_to_json(string_list, tags, tag_type):
    item = {"string": string_list, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    if tag_type == "iobes":
        for char, tag in zip(string_list, tags):
            if tag[0] == "S":
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "E":
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
                entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
    elif tag_type == "iob":
        for char, tag in zip(string_list, tags):
            if tag[0] == "B" or tag[0] == "I":
                if entity_name.strip() == "":
                    entity_start = idx
                entity_name += char
            else:
                if entity_name.strip() != "":
                    item["entities"].append({"word":entity_name, "start": entity_start, "end": idx, type:tag[2:]})
                    entity_name = ""
            idx += 1
    return item

def test_generation(results, path, logger):
    """
    Run perl script to evaluate model
    """
    output_file = os.path.join(path, "generation_predict.utf8")
    with tf.gfile.GFile(output_file, "w") as f:
        for block in results:
            f.write("\n".join(block))
            logger.info("\n".join(block))
