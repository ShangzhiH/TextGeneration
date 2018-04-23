# encoding=utf-8

import tensorflow as tf
import numpy as np
import os
import io

def line_process():
    with io.open("data/dish_data_original.csv", encoding="utf-8") as f1:
        with io.open("data/content.csv", "w", encoding="utf-8") as f2:
            for line in f1:
                split_result = line.split(u"\t")
                if len(split_result) == 2:
                    f2.write(split_result[1])

if __name__ == "__main__":
    line_process()
