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
    parser.add_argument("--part", type=str, required=True,
                            help="partial model dir")

    return parser.parse_args()

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    var_list = tf.contrib.framework.list_variables(FLAGS.model)
    var_part = tf.contrib.framework.list_variables(FLAGS.part)
    var_values, var_dtypes = {}, {}
    var_values_part = {}

    for (name, shape) in var_list:
        var_values[name] = np.zeros(shape)
    for (name, shape) in var_part:
        var_values_part[name] = np.zeros(shape)

    reader = tf.contrib.framework.load_checkpoint(FLAGS.model)
    reader_part = tf.contrib.framework.load_checkpoint(FLAGS.part)
    for name in var_values:
        if name in var_values_part:
            tensor_part = reader_part.get_tensor(name)
            tensor = reader.get_tensor(name)
            print(type(tensor))
            if tensor.equal(tensor_part):
                print('name '+name+' equals')
            else:
                print('name '+name+' is different')
    tf.logging.info("Read from %s and %s", FLAGS.model, FLAGS.part)

if __name__ == "__main__":
    FLAGS = parseargs()
    tf.app.run()
