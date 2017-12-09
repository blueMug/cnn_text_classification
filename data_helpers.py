import numpy as np
import re
import itertools
from collections import Counter
import util


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



def load_data_and_labels(data_file, config, max_length):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    :param data_file:  training set, label <> sententce
    :param config:  describe label, it denotes a one-hot vector, the dimension == number of label
        one label a line. Example:   sports\neducation\nculture   sports:[1,0,0], education:[0,1,0], culture:[0,0,1]
    :param max_length: max sentence length
    :return:
    """

    # Load data from files

    trains = util.read_txt(data_file)
    label_dict = util.read_txt_to_dict(config)
    #
    n_class = len(label_dict)
    one_hot = np.zeros(shape=n_class, dtype=int)
    x_text = []
    y_text = []
    for t in trains:
        line = t.split(' <> ')
        if len(line) < 2:
            continue
        x_text.append(line[1][:max_length])
        label_num = label_dict[line[0].strip()]
        cur_label = np.copy(one_hot)
        cur_label[label_num] = 1
        y_text.append(cur_label)

    # x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split() for s in x_text]

    return [x_text, y_text, n_class]


def pad_sentences(sentences, max_length, padding_word="</s>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = max_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv.insert(0, '<unk>')
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary.get(word, 0) for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]




def load_data(train_file, config, max_length=100, vocabulary=None):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    if train_file == '':
        get_chinese_text()
    sentences, labels, n_class = load_data_and_labels(train_file, config, max_length)
    sentences_padded = pad_sentences(sentences, max_length)
    vocabulary_inv = None
    if vocabulary is None:
        vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, n_class]


def load_test_data(test_file, max_length=100, vocabulary=None):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    contents = util.read_txt(test_file)
    x_text = [line[:max_length] for line in contents]

    # x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split() for s in x_text]
    # Load and preprocess data
    sentences_padded = pad_sentences(x_text, max_length)
    vocabulary = util.read_pickle(vocabulary)
    x = np.array([[vocabulary.get(word, 0) for word in sentence] for sentence in sentences_padded])
    return x, contents


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
