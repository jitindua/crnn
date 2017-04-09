import numpy as np
import cPickle
from collections import defaultdict, Counter, OrderedDict, namedtuple
import sys, re
import pandas as pd

Tokens = namedtuple('Tokens', ['EOS', 'UNK', 'START', 'END', 'ZEROPAD'])
EOS = '+'
encoding='utf8'

def build_data_cv(data_folder, cv=10, clean_string=True, max_word_l=65, n_words=30000, n_chars = 100):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    '''wordcount = Counter()'''
    charcount = Counter()

    tokens = Tokens(
        EOS=EOS,
        UNK='|',    # unk word token
        START='{',  # start-of-word token
        END='}',    # end-of-word token
        ZEROPAD=' ' # zero-pad token
    )

    
    max_word_l_tmp = 0 # max word length of the corpus
    '''idx2word = [tokens.UNK] # unknown word token
    word2idx = OrderedDict()
    word2idx[tokens.UNK] = 0
    idx2char = [tokens.ZEROPAD, tokens.START, tokens.END, tokens.UNK] # zero-pad, start-of-word, end-of-word tokens'''
    char2idx = OrderedDict()
    char2idx[tokens.ZEROPAD] = 0
    char2idx[tokens.START] = 1
    char2idx[tokens.END] = 2
    char2idx[tokens.UNK] = 3

    def update(word):
        '''if word[0] == tokens.UNK:
            if len(word) > 1: # unk token with character info available
                word = word[2:]
        else:
            wordcount.update([word])'''
        word = word.replace(tokens.UNK, '')
        charcount.update(word)

    counts = 0
    with open(pos_file, "rb") as f:
        for line in f:
            rev = []
            line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
            line = line.replace(tokens.START, '')  # start-of-word token is reserved
            line = line.replace(tokens.END, '')  # end-of-word token is reserved
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
                update(word)
                max_word_l_tmp = max(max_word_l_tmp, len(word))
                '''max_word_l_tmp = max(max_word_l_tmp, len(word) + 2) # add 2 for start/end chars
                counts += 1'''
            if tokens.EOS != '':
                update(tokens.EOS)
                '''counts += 1 # PTB uses \n for <eos>, so need to add one more token at the end'''
            datum  = {"y":1,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:
            rev = []
            line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
            line = line.replace(tokens.START, '')  # start-of-word token is reserved
            line = line.replace(tokens.END, '')  # end-of-word token is reserved
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
                max_word_l_tmp = max(max_word_l_tmp, len(word) + 2) # add 2 for start/end chars
                counts += 1
            if tokens.EOS != '':
                update(tokens.EOS)
                counts += 1 # PTB uses \n for <eos>, so need to add one more token at the end
            datum  = {"y":0,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)


    '''print 'Most frequent words:', len(wordcount)
    for ii, ww in enumerate(wordcount.most_common(n_words - 1)):
        word = ww[0]
        word2idx[word] = ii + 1
        idx2word.append(word)
        if ii < 3: print word'''

    print 'Most frequent chars:', len(charcount)
    for ii, cc in enumerate(charcount.most_common(n_chars - 4)):
        char = cc[0]
        char2idx[char] = ii + 4
        '''idx2char.append(char)'''
        if ii < 3: print char

    print 'Char counts:'
    for ii, cc in enumerate(charcount.most_common()):
        print ii, cc[0].encode(encoding), cc[1]

    print 'After first pass of data, max word length is: ', max_word_l_tmp

    # if actual max word length is less than the limit, use that
    max_word_l = min(max_word_l_tmp, max_word_l)

    '''# Preallocate the tensors we will need.
    # Watch out the second one needs a lot of RAM.
    output_tensor = np.empty(split_counts[counts], dtype='int32')
    output_chars = np.zeros((split_counts[counts], max_word_l), dtype='int32')

    def append(word, word_num):
        chars = [char2idx[tokens.START]] # start-of-word symbol
        if word[0] == tokens.UNK and len(word) > 1: # unk token with character info available
            word = word[2:]
            output_tensor[word_num] = word2idx[tokens.UNK]
        else:
            output_tensor[word_num] = word2idx[word] if word in word2idx else word2idx[tokens.UNK]
        chars += [char2idx[char] for char in word if char in char2idx]
        chars.append(char2idx[tokens.END]) # end-of-word symbol
        if len(chars) >= max_word_l:
            chars[max_word_l-1] = char2idx[tokens.END]
            output_chars[word_num] = chars[:max_word_l]
        else:
            output_chars[word_num, :len(chars)] = chars
        return word_num + 1       
        
    with open(pos_file, "rb") as f:
        for line in f:
            rev = []
            line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
            line = line.replace(tokens.START, '')  # start-of-word token is reserved
            line = line.replace(tokens.END, '')  # end-of-word token is reserved
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                word_num = append(word, word_num)
            if tokens.EOS != '':   # PTB does not have <eos> so we add a character for <eos> tokens
                word_num = append(tokens.EOS, word_num)   # other datasets don't need this

    with open(neg_file, "rb") as f:
        for line in f:
            rev = []
            line = line.replace('<unk>', tokens.UNK)  # replace unk with a single character
            line = line.replace(tokens.START, '')  # start-of-word token is reserved
            line = line.replace(tokens.END, '')  # end-of-word token is reserved
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                word_num = append(word, word_num)
            if tokens.EOS != '':   # PTB does not have <eos> so we add a character for <eos> tokens
                word_num = append(tokens.EOS, word_num)   # other datasets don't need this

    all_data = output_tensor
    all_data_char = output_chars

    vocab_size = len(idx2word)
    char_vocab_size = len(idx2char)'''
    
    '''return revs, vocab, wordcount, charcount, max_word_l, idx2word, word2idx, idx2char, char2idx'''
    return revs, vocab, charcount, max_word_l, char2idx, tokens

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec for words in vocab
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
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
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

if __name__=="__main__":
    w2v_file = sys.argv[1]
    data_folder = ["./data/rt-data/rt-polarity.pos",
                   "./data/rt-data/rt-polarity.neg"]
    print "loading data...",
    # tokenized sentences and y
    #revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    '''revs, vocab, wordcount, charcount, max_word_l, idx2word, word2idx, idx2char, char2idx = build_data_cv(data_folder, max_word_l, n_words, n_chars, cv=10, clean_string=True)'''
    revs, vocab, charcount, max_word_l, char2idx, tokens = build_data_cv(data_folder, cv=10, clean_string=True, max_word_l=65, n_words=30000, n_chars = 100)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    # google vector for words in both google w2v and vocab
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    # random vector for words in vocab but not in google w2v
    add_unknown_words(w2v, vocab)
    # idx2vec, word2idx mappings
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    # random vectors for all words in vocab
    add_unknown_words(rand_vecs, vocab)
    # index to random vectors
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    '''cPickle.dump([wordcount, charcount, max_word_l, idx2word, word2idx, idx2char, char2idx], open("mr2.p", "wb"))'''
    cPickle.dump([len(charcount), max_word_l, char2idx], open("mr2.p", "wb"))
    
    #print ('charcount', charcount, '\n\nmax_word_l', max_word_l, '\n\nchar2idx', char2idx, '\n\n')
    print "dataset created!"


