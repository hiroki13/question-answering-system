import sys
import time
import numpy as np

import io_utils
from preprocessor import corpus_statistics, sample_format, theano_format
from model_builder import set_model, set_train_f


def train(argv):
    print 'SETTING UP A TRAINING SETTING'

    emb = None
    batch_size = argv.batch_size
    window = argv.window

    ##############
    # LOAD FILES #
    ##############

    """ Load files """
    # corpus: 1D: n_sents, 2D: n_words, 3D: (word, pas_info, pas_id)
    corpus, vocab_word = io_utils.load(argv.train_data)
    corpus_statistics(corpus)

    ##############
    # PREPROCESS #
    ##############

    """ Preprocessing """
    # samples: 1D: n_sents, 2D: [word_ids, tag_ids, prd_indices, contexts]
    samples = sample_format(corpus, vocab_word)

    # dataset = [tr_x, tr_y, tr_l]
    # tr_x=features: 1D: n_samples * n_words, 2D: window; elem=word id
    # tr_y=labels: 1D: n_samples; elem=scalar
    # tr_l=question length: 1D: n_samples * 2; elem=scalar
    # bb_x=batch indices for x: 1D: n_samples / batch_size + 1; elem=(bob, eob)
    # bb_y=batch indices for y: 1D: n_samples / batch_size + 1; elem=(bob, eob)
    tr_dataset, bb_x, bb_y = theano_format(samples, batch_size)

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    """ Set a model """
    print '\n\nBuilding a model...'
    model = set_model(argv=argv, emb=emb, vocab=vocab_word)
    train_f = set_train_f(model, tr_dataset)

    ###############
    # TRAIN MODEL #
    ###############

    print '\nTraining start\n'
    indices = range(len(bb_y))
    n_samples = len(samples)

    for epoch in xrange(argv.epoch):
        print '\nEpoch: %d' % (epoch + 1)
        print '\tTRAIN\n\t',

        np.random.shuffle(indices)
        start = time.time()

        ttl_nll = 0.
        ttl_crr = 0.
        for i, b_index in enumerate(indices):
            if (i + 1) % 100 == 0:
                print i + 1,
                sys.stdout.flush()

            bb_x_i = bb_x[b_index]
            bb_y_i = bb_y[b_index]
            crr, nll = train_f(index=b_index, bob_x=bb_x_i[0], eob_x=bb_x_i[1], bob_y=bb_y_i[0], eob_y=bb_y_i[1])
            ttl_crr += np.sum(crr)
            ttl_nll += nll

        end = time.time()
        print '\n\tTime: %f\tNLL: %f' % ((end - start), ttl_nll)
        print '\tACC: %f  CRR: %d   TOTAL: %d' % (ttl_crr/n_samples, ttl_crr, n_samples)


def main(argv):
    train(argv)

