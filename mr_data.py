import cPickle
import numpy as np
from collections import namedtuple
from process_mr_data import Tokens
Tokens = namedtuple('Tokens', ['EOS', 'UNK', 'START', 'END', 'ZEROPAD'])
EOS = '+'

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5, pad_left=True):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    if pad_left:
        for i in xrange(pad):
            x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x


def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5, pad_left=True):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h, pad_left=pad_left)
        sent.append(rev["y"])
        if rev["split"]==cv:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]


def get_char_idx_from_sent(sent, max_word_l, char2idx, tokens, max_l=51, k=300, filter_h=5, pad_left=True):
    """
    Transforms sentence into a list of list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    if pad_left:
        for i in xrange(pad):
            #temp = np.full(max_word_l, char2idx[tokens.ZEROPAD])
            temp = []
            for char in xrange(max_word_l):
                temp.append(char2idx[tokens.ZEROPAD])
            x.append(temp)

    words = sent.split()
    for word in words:
        temp = [char2idx[tokens.START]]
        for char in word:
            if char in char2idx:
                temp.append(char2idx[char])
            else:
                temp.append(char2idx[tokens.UNK])
        if len(word) >= max_word_l:
            temp[max_word_l-1] = char2idx[tokens.END]
            temp = temp[:max_word_l]
        else:
            temp.append(char2idx[tokens.END])
            while len(temp) < max_word_l:
                temp.append(char2idx[tokens.ZEROPAD])
        x.append(temp)
    maxlen = max_l+2*pad
    if len(x) >= maxlen:
        x = x[:maxlen]
    else:
        while len(x) < maxlen:
            temp = []
            for chars in xrange(max_word_l):
                temp.append(char2idx[tokens.ZEROPAD])
            x.append(temp)
            #x.append(np.full(max_word_l, char2idx[tokens.ZEROPAD]))
    return x


def make_char_idx_data_cv(revs, cv, max_word_l, char2idx, tokens, max_l=51, k=300, filter_h=5, pad_left=True):
    """
    Transforms sentences into a 3-d matrix.
    """
    train, test = [], []
    trainy, testy = [], []
    for rev in revs:
        sent = get_char_idx_from_sent(rev["text"], max_word_l, char2idx, tokens, max_l, k, filter_h, pad_left=pad_left)
        if rev["split"]==cv:
            test.append(sent)
            testy.append(rev["y"])
        else:
            train.append(sent)
            trainy.append(rev["y"])
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    trainy = np.array(trainy,dtype="int")
    testy = np.array(testy,dtype="int")
    return [train, trainy, test, testy]


def make_idx_data_cv_org_text(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5, pad_left=True):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        if rev["split"]==cv:
            test.append(rev["text"])
        else:
            train.append(rev["text"])
    return [train, test]


x = None
y = None
def load_data(fold, pad_left=True, word_model = True):
    global x
    global y
    if x is None:
        x = cPickle.load(open("mr.p","rb"))
    if y is None:
        y = cPickle.load(open("mr2.p","rb"))

    tokens = Tokens(
        EOS=EOS,
        UNK='|',    # unk word token
        START='{',  # start-of-word token
        END='}',    # end-of-word token
        ZEROPAD=' ' # zero-pad token
    )

    # revs: tokenized sentences and y
    # W: idx2vec
    # W2: idx2 random vec
    # word_idx_map: word2idx
    # vocab: vocab of words
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    charcount, max_word_l, char2idx, idx2char = y[0], y[1], y[2], y[3]
    if word_model:
        datasets = make_idx_data_cv(revs, word_idx_map, fold, max_l=56, k=300, filter_h=5, pad_left=pad_left)
        img_h = len(datasets[0][0])-1
        return datasets[0][:,:img_h], datasets[0][:, -1], datasets[1][:,: img_h], datasets[1][: , -1], W, W2
    else:
        datasets_char = make_char_idx_data_cv(revs, fold, max_word_l, char2idx, tokens, max_l=56, k=300, filter_h=5, pad_left=pad_left)

        '''for sent in datasets_char[0][:2]:
            print '#####'
            for word in sent:
                print word
                for char in word:
                    if char:
                        print idx2char[char]
        print 'y::'
        for y in datasets_char[1][:2]:
            print y'''
        return datasets_char[0], datasets_char[1], datasets_char[2], datasets_char[3], charcount, max_word_l

# TBD
def load_data_org(fold, pad_left=True):
    global x
    if x is None:
        x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    datasets = make_idx_data_cv(revs, word_idx_map, fold, max_l=56, k=300, filter_h=5, pad_left=pad_left)
    train_text, test_text = make_idx_data_cv_org_text(revs, word_idx_map, fold, max_l=56, k=300, filter_h=5, pad_left=pad_left)
    img_h = len(datasets[0][0])-1
    return datasets[0][:,:img_h], datasets[0][:, -1], datasets[1][:,: img_h], datasets[1][: , -1], W, W2, train_text, test_text
