#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import operator
import os

import numpy as np
import tensorflow as tf


def parseargs():
    msg = "Average checkpoints"
    usage = "average.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    parser.add_argument("--model", type=str, required=True,
                        help="checkpoint dir")

    return parser.parse_args()

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    var_list = tf.contrib.framework.list_variables(FLAGS.model)
    var_values, var_dtypes = {}, {}
    model_from = "transformer_cov"
    model_to = "transformer_lrp"

    count = 0
    for (name, shape) in var_list:
        if True:#not name.startswith("global_step") and not 'Adam' in name:
            count += 1
            print(name, shape)
            name = name.replace(model_from, model_to)
            var_values[name] = np.zeros(shape)
    print(len(var_list))
    print(count) 


if __name__ == "__main__":
    FLAGS = parseargs()
    tf.app.run()
