import pickle


def read_txt(filename):
    res = list()
    with open(filename, 'rb') as f:
        for line in f:
            res.append(line.decode('utf-8').strip())
    return res


def read_txt_to_dict(filename):
    labels = read_txt(filename)
    #
    label_dict = {}
    for i, label in enumerate(labels):
        label_dict[label] = i
    return label_dict


def save_to_txt(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))


def save_to_pickle(filename, content):
    with open(filename, 'wb') as f:
        pickle.dump(content, f)


def read_pickle(filename):
    pkl_file = open(filename, 'rb')
    data = pickle.load(pkl_file)
    return data

# read_pickle('data/vocab.pkl')