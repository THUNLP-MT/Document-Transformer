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
    parser.add_argument("--output", type=str, help="output path")

    return parser.parse_args()

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    var_list = tf.contrib.framework.list_variables(FLAGS.model)
    var_values, var_dtypes = {}, {}
    model_from = "transformer"
    model_to = "contextual_transformer"

    for (name, shape) in var_list:
        if True:#not name.startswith("global_step") and not 'Adam' in name:
            name = name.replace(model_from, model_to)
            var_values[name] = np.zeros(shape)
            print(name)

    reader = tf.contrib.framework.load_checkpoint(FLAGS.model)
    for name in var_values:
        name_ori = name.replace(model_to, model_from)
        tensor = reader.get_tensor(name_ori)
        var_dtypes[name] = tensor.dtype
        var_values[name] += tensor
    tf.logging.info("Read from %s", FLAGS.model)

    tf_vars = [
        tf.get_variable(name, shape=var_values[name].shape,
                        dtype=var_dtypes[name]) for name in var_values
    ]
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    global_step = tf.Variable(0, name="global_step", trainable=False,
                              dtype=tf.int64)
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                               var_values.iteritems()):
            sess.run(assign_op, {p: value})
        saved_name = os.path.join(FLAGS.output, "new")
        saver.save(sess, saved_name, global_step=global_step)

    tf.logging.info("Averaged checkpoints saved in %s", saved_name)

if __name__ == "__main__":
    FLAGS = parseargs()
    tf.app.run()
