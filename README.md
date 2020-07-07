# Improving the Transformer Translation Model with Document-Level Context
## Contents
* [Introduction](#introduction)
* [Usage](#usage)
* [Citation](#citation)
* [FAQ](#faq)

## Introduction

This is the implementation of our work, which extends Transformer to integrate document-level context \[[paper](https://arxiv.org/abs/1810.03581)\]. The implementation is on top of [THUMT](https://github.com/thumt/THUMT)

## Usage

Note: The usage is not user-friendly. May improve later.

1. Train a standard Transformer model, please refer to the user manual of [THUMT](https://github.com/thumt/THUMT). Suppose that model_baseline/model.ckpt-30000 performs best on validation set.

2. Generate a dummy improved Transformer model with the following command:

<pre><code>python THUMT/thumt/bin/trainer_ctx.py --inputs [source corpus] [target corpus] \
                                      --context [context corpus] \
                                      --vocabulary [source vocabulary] [target vocabulary] \
                                      --output model_dummy --model contextual_transformer \
                                      --parameters train_steps=1
</code></pre>

3. Generate the initial model by merging the standard Transformer model into the dummy model, then create a checkpoint file:

<pre><code>python THUMT/thumt/scripts/combine_add.py --model model_dummy/model.ckpt-0 \
                                         --part model_baseline/model.ckpt-30000 --output train
printf 'model_checkpoint_path: "new-0"\nall_model_checkpoint_paths: "new-0"' > train/checkpoint
</code></pre>


4. Train the improved Transformer model with the following command:

<pre><code>python THUMT/thumt/bin/trainer_ctx.py --inputs [source corpus] [target corpus] \
                                      --context [context corpus] \
                                      --vocabulary [source vocabulary] [target vocabulary] \
                                      --output train --model contextual_transformer \
                                      --parameters start_steps=30000,num_context_layers=1
</code></pre>

5. Translate with the improved Transformer model:

<pre><code>python THUMT/thumt/bin/translator_ctx.py --inputs [source corpus] --context [context corpus] \
                                         --output [translation result] \
                                         --vocabulary [source vocabulary] [target vocabulary] \
                                         --model contextual_transformer --checkpoints [model path] \
                                         --parameters num_context_layers=1
</code></pre>

## Citation

Please cite the following paper if you use the code:

<pre><code>@InProceedings{Zhang:18,
  author    = {Zhang, Jiacheng and Luan, Huanbo and Sun, Maosong and Zhai, Feifei and Xu, Jingfang and Zhang, Min and Liu, Yang},
  title     = {Improving the Transformer Translation Model with Document-Level Context},
  booktitle = {Proceedings of EMNLP},
  year      = {2018},
}
</code></pre>


## FAQ

1. What is the context corpus?

The context corpus file contains one context sentence each line. Normally, context sentence is the several preceding source sentences within a document. For example, if the origin document-level corpus is:

<pre><code>==== source ====
&lt;document id=XXX>
&lt;seg id=1>source sentence #1&lt;/seg>
&lt;seg id=2>source sentence #2&lt;/seg>
&lt;seg id=3>source sentence #3&lt;/seg>
&lt;seg id=4>source sentence #4&lt;/seg>
&lt;/document>

==== target ====
&lt;document id=XXX>
&lt;seg id=1>target sentence #1&lt;/seg>
&lt;seg id=2>target sentence #2&lt;/seg>
&lt;seg id=3>target sentence #3&lt;/seg>
&lt;seg id=4>target sentence #4&lt;/seg>
&lt;/document></code></pre>

The inputs to our system should be processed as (suppose that 2 preceding source sentences are used as context):

<pre><code>==== train.src ==== (source corpus)
source sentence #1
source sentence #2
source sentence #3
source sentence #4

==== train.ctx ==== (context corpus)
(the first line is empty)
source sentence #1
source sentence #1 source sentence #2 (there is only a space between the two sentence)
source sentence #2 source sentence #3

==== train.trg ==== (target corpus)
target sentence #1
target sentence #2
target sentence #3
target sentence #4</code></pre>



