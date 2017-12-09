import mxnet as mx
import argparse
import logging
import data_helpers
import util

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="CNN for text classification evaluation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test', type=str, default='',
                    help='test set')
parser.add_argument('--output', type=str, default='',
                    help='output file, if null, output at the console')
parser.add_argument('--config', type=str, default='',
                    help='config file, denote labels')
parser.add_argument('--vocab', type=str, default='./data/vocab.pkl',
                    help='vocab file path for generation')
parser.add_argument('--model-name', type=str, default='checkpoint',
                    help='model name')
parser.add_argument('--max_length', type=int, default=100,
                    help='max sentence length')
parser.add_argument('--gpus', type=str, default='',
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ')
parser.add_argument('--checkpoint', type=int, default=0,
                    help='max num of epochs')
parser.add_argument('--batch_size', type=int, default=1,
                    help='the batch size.')


def data_construct(input_file, batch_size, vocab, max_length=100):
    logger.info('Loading data...')

    x_test, contents = data_helpers.load_test_data(input_file, max_length, vocab)
    test_iter = mx.io.NDArrayIter(x_test, None, batch_size)
    return test_iter, contents


if __name__ == '__main__':
    # parse args
    args = parser.parse_args()

    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix='checkpoint/'+args.model_name, epoch=args.checkpoint)

    test_for_predict, contents = data_construct(input_file=args.test, batch_size=args.batch_size, vocab=args.vocab,
                                                max_length=args.max_length)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=[('data', (args.batch_size, args.max_length))])
    mod.set_params(arg_params, aux_params, allow_missing=True)

    prob = mod.predict(test_for_predict)
    labels = prob.argmax(axis=1)
    label_name = util.read_txt(args.config)

    result = []
    for label, sentence in zip(labels, contents):
        result.append(label_name[int(label.asscalar())] + ' <> ' + sentence)
    output = args.output
    if output == '':
        print(result)
    else:
        util.save_to_txt(output, result)
    logger.info('finished...')
