# -*- coding: utf-8 -*-

import gzip
from collections import defaultdict

from vocab import Vocab


PAD = u'<PAD>'
UNK = u'<UNK>'


def load(path, vocab_word=Vocab()):
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

    for w, f in sorted(word_freqs.items(), key=lambda (k, v): -v):
        vocab_word.add_word(w)

    return corpus, vocab_word


def load_sep(path, label):
    corpus = []

    with gzip.open(path) as f:
        sample = []
        for line in f:
            line = line.rstrip().split('\t')

            if len(line) != 3:
                corpus.append(sample)
                sample = []
            else:
                sample.append((line, label))

    return corpus


def save_sep(fn, data):
    """
    :param fn: string; file name
    :param data: 1D: n_qa; elem: QA
    """
    print 'Save %s' % fn

    with gzip.open(fn + '.gz', 'wb') as gf:
        for i, sample in enumerate(data):
            if len(sample) != 2:
                continue

            flag = True
            q_text = 'Sample-%d\t%d\n' % (i+1, sample[0][1])
            for q in sample:
                if len(q[0][0]) == 0 or len(q[0][1]) == 0 or len(q[0][2]) == 0:
                    flag = False
                    break
                q_text += '%s\t%s\t%s\n' % (q[0][0], q[0][1], q[0][2])
            q_text += '\n'

            if flag:
                gf.writelines(q_text)
