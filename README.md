Implementing  CNN Text Classification in MXNet
============

Recently, I have been learning mxnet for Natural Language Processing (NLP). I followed this official code in [MXNet github](https://github.com/apache/incubator-mxnet/tree/master/example/cnn_chinese_text_classification).
However, I find the official codes are too simple to run a whole process, so I changed it.

## The main difference with the official version
- Inference code were added, one can use his trained model to do prediction
- The MXNet version is 0.12.1, so some original functions may be deprecated
- Binary classification tasks were changed to multi-category tasks
- The codes about pretrained embedding were removed, data format were changed
- Label shape were changed to (batch_size,)

## Data
#### training and validation data
two txt file, the format of each line is: \<label> sentence.

- \<pos> This is the best movie about troubled teens since 1998's whatever.
- \<neg> This 10th film in the series looks and feels tired.

#### config data
one label a line, the number of labels is equals to total classes.
- pos
- neg

#### inference data
one sentence a line, without \<label>

#### inference data with evaluation
the format of each line is: \<label> sentence, like validation file

The data is recommended to be tokenized or segmented(Chinese).

## Quick start
``python cnn_model.py --train path/to/train.data --validate /path/to/validate.data --config /path/to/config``

``python inference.py --test python/to/inference.data --config /path/to/config --checkpoint 1``

``python inference.py --test python/to/inference-evaluation.data --config /path/to/config --checkpoint 1 --evaluation``

## References
- [Implementing CNN + Highway Network for Chinese Text Classification in MXNet](https://github.com/apache/incubator-mxnet/tree/master/example/cnn_chinese_text_classification)
- ["Implementing a CNN for Text Classification in Tensorflow" blog post.](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)

