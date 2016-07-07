import sys
import time
import numpy as np

import io_utils
from preprocessor import corpus_statistics, sample_format, theano_format
from model_builder import set_predict_f, set_rank_f


def test(argv):
    print '\nSETTING UP A TEST SETTING\n'

    task = argv.task
    batch_size = argv.batch_size
    window = argv.window

    print '\tTASK: %s\tBATCH: %d\tWINDOW: %d' % (task, batch_size, window)

    ##############
    # LOAD FILES #
    ##############

    """ Load files """
    # corpus: 1D: n_sents, 2D: n_words, 3D: (word, pas_info, pas_id)
    vocab_word = io_utils.load_data(argv.load_vocab)

    if argv.dev_data:
        dev_corpus, _ = io_utils.load(argv.dev_data, vocab_word, False)
        print '\nDEV CORPUS'
        corpus_statistics(dev_corpus)

    if argv.test_data:
        test_corpus, _ = io_utils.load(argv.test_data, vocab_word, False)
        print '\nTEST CORPUS'
        corpus_statistics(test_corpus)

    print '\nVocab: %d' % vocab_word.size()

    ##############
    # PREPROCESS #
    ##############

    """ Preprocessing """
    # samples: 1D: n_sents, 2D: [word_ids, tag_ids, prd_indices, contexts]
    if argv.dev_data:
        dev_samples = sample_format(dev_corpus, vocab_word, window)
        n_dev_samples = len(dev_samples)

    if argv.test_data:
        test_samples = sample_format(test_corpus, vocab_word, window)
        n_te_samples = len(test_samples)

    # dataset = [x, y, l]
    # x=features: 1D: n_samples * n_words, 2D: window; elem=word id
    # y=labels: 1D: n_samples; elem=scalar
    # l=question length: 1D: n_samples * 2; elem=scalar
    # bb_x=batch indices for x: 1D: n_samples / batch_size + 1; elem=(bob, eob)
    # bb_y=batch indices for y: 1D: n_samples / batch_size + 1; elem=(bob, eob)

    if argv.dev_data:
        dev_dataset, dev_bb_x, dev_bb_y = theano_format(dev_samples, batch_size)

    if argv.test_data:
        te_dataset, te_bb_x, te_bb_y = theano_format(test_samples, batch_size)

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    """ Set a model """
    print '\n\nBuilding a model...'
    model = io_utils.load_data(argv.load_model)

    if argv.dev_data:
        if task == 'binary':
            dev_f = set_predict_f(model, dev_dataset)
        elif task == 'ranking':
            dev_f = set_rank_f(model, dev_dataset)
    if argv.test_data:
        if task == 'binary':
            test_f = set_predict_f(model, te_dataset)
        elif task == 'ranking':
            test_f = set_rank_f(model, te_dataset)

    ########
    # TEST #
    ########

    if argv.dev_data:
        print '\n\tDEV\n\t',
        predict(dev_f, dev_bb_x, dev_bb_y, n_dev_samples)

    if argv.test_data:
        print '\n\tTEST\n\t',
        predict(test_f, te_bb_x, te_bb_y, n_te_samples)


def predict(f, bb_x, bb_y, n_samples):
    ttl_crr = 0.
    start = time.time()
    for i in xrange(len(bb_y)):
        if (i + 1) % 100 == 0:
            print i + 1,
            sys.stdout.flush()

        bb_x_i = bb_x[i]
        bb_y_i = bb_y[i]
        crr = f(index=i, bob_x=bb_x_i[0], eob_x=bb_x_i[1], bob_y=bb_y_i[0], eob_y=bb_y_i[1])
        ttl_crr += np.sum(crr)

    end = time.time()
    acc = ttl_crr/n_samples
    print '\n\tTime: %f' % (end - start)
    print '\tACC: %f  CRR: %d   TOTAL: %d' % (acc, ttl_crr, n_samples)


def main(argv):
    test(argv)

