import sys
import os
import mxnet as mx
import numpy as np
import argparse
import logging

import time

from custom_init import CustomInit

import data_helpers
import util

parser = argparse.ArgumentParser(description="CNN for text classification",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train', type=str, default='',
                    help='train set')
parser.add_argument('--validate', type=str, default='',
                    help='validate set')
parser.add_argument('--config', type=str, default='',
                    help='config file, denote labels')
parser.add_argument('--vocab', type=str, default='./data/vocab.pkl',
                    help='vocab file path for generation')
parser.add_argument('--model-name', type=str, default='checkpoint',
                    help='model name')
parser.add_argument('--num-embed', type=int, default=300,
                    help='embedding layer size')
parser.add_argument('--max_length', type=int, default=100,
                    help='max sentence length')
parser.add_argument('--gpus', type=str, default='',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ')
parser.add_argument('--kv-store', type=str, default='local',
                    help='key-value store type')
parser.add_argument('--num-epochs', type=int, default=150,
                    help='max num of epochs')
parser.add_argument('--batch-size', type=int, default=100,
                    help='the batch size.')
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    help='the optimizer type')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='initial learning rate')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout rate')
parser.add_argument('--disp-batches', type=int, default=50,
                    help='show progress for every n batches')
parser.add_argument('--save-period', type=int, default=10,
                    help='save checkpoint for every n epochs')
parser.add_argument('--log-name', type=str, default='./cnn_text_classification.log',
                    help='log file path')
# parse args
args = parser.parse_args()

fmt = '%(asctime)s:filename %(filename)s: lineno %(lineno)d:%(levelname)s:%(message)s'
logging.basicConfig(format=fmt, filemode='a+', filename=args.log_name, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def save_model(checkname):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    return mx.callback.do_checkpoint("checkpoint/"+checkname, args.save_period)


def highway(data):
    _data = data
    high_weight = mx.sym.Variable('high_weight')
    high_bias = mx.sym.Variable('high_bias')
    high_fc = mx.sym.FullyConnected(data=data, weight=high_weight, bias=high_bias, num_hidden=300, name='high_fc')
    high_relu = mx.sym.Activation(high_fc, act_type='relu')

    high_trans_weight = mx.sym.Variable('high_trans_weight')
    high_trans_bias = mx.sym.Variable('high_trans_bias')
    high_trans_fc = mx.sym.FullyConnected(data=_data, weight=high_trans_weight, bias=high_trans_bias, num_hidden=300,
                                          name='high_trans_sigmoid')
    high_trans_sigmoid = mx.sym.Activation(high_trans_fc, act_type='sigmoid')

    return high_relu * high_trans_sigmoid + _data * (1 - high_trans_sigmoid)


def data_iter(train_file, validate_file, config, batch_size, num_embed, vocab_path, max_length=100):
    logger.info('Loading data...')

    x_train, y_train, vocab, vocab_inv, n_class = data_helpers.load_data(train_file, config, max_length, None)
    embed_size = num_embed
    sentence_size = x_train.shape[1]
    vocab_size = len(vocab)
    util.save_to_pickle(vocab_path, vocab)
    x_dev, y_dev, _, _, _ = data_helpers.load_data(validate_file, config, max_length, vocab)

    # randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    logger.info('Train/Valid split: %d/%d' % (len(y_train), len(y_dev)))
    logger.info('train shape: %(shape)s', {'shape': x_train.shape})
    logger.info('valid shape: %(shape)s', {'shape': x_dev.shape})
    logger.info('sentence max words: %(shape)s', {'shape': sentence_size})
    logger.info('embedding size: %(msg)s', {'msg': embed_size})
    logger.info('vocab size: %(msg)s', {'msg': vocab_size})

    train = mx.io.NDArrayIter(
        x_train, y_train, batch_size, shuffle=True)
    valid = mx.io.NDArrayIter(
        x_dev, y_dev, batch_size)
    return train, valid, sentence_size, embed_size, vocab_size, n_class


def sym_gen(sentence_size, num_embed, vocab_size,
            num_label=2, filter_list=[3, 4, 5], num_filter=100,
            dropout=0.0):
    input_x = mx.sym.Variable('data')
    input_y = mx.sym.Variable('softmax_label')

    # embedding layer
    embed_layer = mx.sym.Embedding(data=input_x, input_dim=vocab_size, output_dim=num_embed, name='vocab_embed')
    conv_input = mx.sym.reshape(data=embed_layer, shape=(-1, 1, sentence_size, num_embed))

    # create convolution + (max) pooling layer for each filter operation
    pooled_outputs = []
    for i, filter_size in enumerate(filter_list):
        convi = mx.sym.Convolution(data=conv_input, kernel=(filter_size, num_embed), num_filter=num_filter)
        relui = mx.sym.Activation(data=convi, act_type='relu')
        pooli = mx.sym.Pooling(data=relui, pool_type='max', kernel=(sentence_size - filter_size + 1, 1), stride=(1, 1))
        pooled_outputs.append(pooli)

    # combine all pooled outputs
    total_filters = num_filter * len(filter_list)
    concat = mx.sym.concat(*pooled_outputs, dim=1)
    h_pool = mx.sym.reshape(data=concat, shape=(-1, total_filters))

    # highway network
    h_pool = highway(h_pool)

    # dropout layer
    if dropout > 0.0:
        h_drop = mx.sym.Dropout(data=h_pool, p=dropout)
    else:
        h_drop = h_pool

    # fully connected
    cls_weight = mx.sym.Variable('cls_weight')
    cls_bias = mx.sym.Variable('cls_bias')

    fc = mx.sym.FullyConnected(data=h_drop, weight=cls_weight, bias=cls_bias, num_hidden=num_label)

    # softmax output
    sm = mx.sym.SoftmaxOutput(data=fc, label=input_y, name='softmax')

    return sm, ('data',), ('softmax_label',)


def train(symbol, train_iter, valid_iter, data_names, label_names, checkname):
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    module = mx.mod.Module(symbol, data_names=data_names, label_names=label_names, context=devs)

    init_params = {
        'vocab_embed_weight': {'uniform': 0.1},
        'convolution0_weight': {'uniform': 0.1}, 'convolution0_bias': {'costant': 0},
        'convolution1_weight': {'uniform': 0.1}, 'convolution1_bias': {'costant': 0},
        'convolution2_weight': {'uniform': 0.1}, 'convolution2_bias': {'costant': 0},
        'high_weight': {'uniform': 0.1}, 'high_bias': {'costant': 0},
        'high_trans_weight': {'uniform': 0.1}, 'high_trans_bias': {'costant': -2},
        'cls_weight': {'uniform': 0.1}, 'cls_bias': {'costant': 0},
    }
    # custom init_params
    module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    module.init_params(CustomInit(init_params))
    lr_sch = mx.lr_scheduler.FactorScheduler(step=25000, factor=0.999)
    module.init_optimizer(
        optimizer='rmsprop', optimizer_params={'learning_rate': 0.0005, 'lr_scheduler': lr_sch})

    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)
    mon = mx.mon.Monitor(25000, norm_stat)

    module.fit(train_data=train_iter,
               eval_data=valid_iter,
               eval_metric='acc',
               kvstore=args.kv_store,
               monitor=mon,
               num_epoch=args.num_epochs,
               batch_end_callback=mx.callback.Speedometer(args.batch_size, args.disp_batches),
               epoch_end_callback=save_model(checkname))


if __name__ == '__main__':

    # data iter
    train_iter, valid_iter, sentence_size, embed_size, vocab_size, n_class = data_iter(
        args.train, args.validate, args.config, args.batch_size, args.num_embed, args.vocab, args.max_length)

    # network symbol
    symbol, data_names, label_names = sym_gen(sentence_size,
                                              embed_size,
                                              vocab_size,
                                              num_label=n_class, filter_list=[3, 4, 5], num_filter=100,
                                              dropout=args.dropout)
    # train cnn model
    train(symbol, train_iter, valid_iter, data_names, label_names, args.model_name)
