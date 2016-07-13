# -*- coding: utf-8 -*-

import gzip
import cPickle
from collections import defaultdict

from vocab import Vocab


PAD = u'<PAD>'
UNK = u'<UNK>'


def load(path, vocab_word=Vocab(), register=True):
    corpus = []
    word_freqs = defaultdict(int)

    if vocab_word.size() == 0:
        vocab_word.add_word(PAD)
        vocab_word.add_word(UNK)

    with gzip.open(path) as f:
        sample = []
        label = -1
        for line in f:
            line = line.rstrip().split('\t')

            if line[0].startswith('Sample'):
                label = int(line[-1])
            elif len(line) != 3:
                corpus.append((sample, label))
                sample = []
            else:
                title = [w.lower() for w in line[1].split()]
                body = [w.lower() for w in line[2].split()]

                for w in title:
                    word_freqs[w] += 1
                for w in body:
                    word_freqs[w] += 1

                sample.append((title, body))

    if register:
        for w, f in sorted(word_freqs.items(), key=lambda (k, v): -v):
            vocab_word.add_word(w)

    return corpus, vocab_word


def dump_data(data, fn):
    with gzip.open(fn + '.pkl.gz', 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def load_data(fn):
    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)

