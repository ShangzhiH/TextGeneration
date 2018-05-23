# encoding=utf-8

import argparse

from model_usage_dataset import ModelUsage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clean_map",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--clean",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--use_train_sampling",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--train_sample_prob",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--char_dim",
        type=int,
        default=200
    )
    parser.add_argument(
        "--hidden_dim",
        type=float,
        default=200
    )
    parser.add_argument(
        "--rnn_layer_num",
        type=int,
        default=2
    )
    parser.add_argument(
        "--infer_num",
        type=int,
        default=3
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=5.0
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )
    parser.add_argument(
        ## 是否用0代替所有数字
        "--zeros",
        type=bool,
        default=True
    )
    parser.add_argument(
        ## 是否全部转小写
        "--lower",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=100
    )
    parser.add_argument(
        "--steps_check",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--steps_eval",
        type=int,
        default=10,
    )
    parser.add_argument(
        ## 存放文件的根目录
        "--root_path",
        type=str,
        default=""
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="ckpt/",
    )
    parser.add_argument(
        "--best_ckpt_path",
        type=str,
        default="best_ckpt/"
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="summary/",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="log/",
    )
    parser.add_argument(
        "--map_path",
        type=str,
        default="maps",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="vocab/",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="result/",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="data/"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="weicheng_train.txt",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default="ner_data_dev",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="ner_data_dev",
    )

    FLAGS, unparsed = parser.parse_known_args()
    assert FLAGS.clip < 5.1, "gradient clip should't be too much"
    assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
    assert FLAGS.lr > 0, "learning rate must larger than zero"

    usage = ModelUsage(FLAGS)
    usage.run()


