import sys
import time
import numpy as np

import io_utils
from preprocessor import corpus_statistics, sample_format, theano_format
from model_builder import set_model, set_train_f, set_predict_f


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
    tr_corpus, vocab_word = io_utils.load(argv.train_data)
    print 'TRAIN CORPUS'
    corpus_statistics(tr_corpus)

    if argv.dev_data:
        dev_corpus, vocab_word = io_utils.load(argv.dev_data, vocab_word)
        print 'DEV CORPUS'
        corpus_statistics(dev_corpus)

    if argv.test_data:
        test_corpus, vocab_word = io_utils.load(argv.test_data, vocab_word)
        print 'TEST CORPUS'
        corpus_statistics(test_corpus)

    print '\nVocab: %d' % vocab_word.size()

    ##############
    # PREPROCESS #
    ##############

    """ Preprocessing """
    # samples: 1D: n_sents, 2D: [word_ids, tag_ids, prd_indices, contexts]
    train_samples = sample_format(tr_corpus, vocab_word, window)
    n_tr_samples = len(train_samples)

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
    tr_dataset, tr_bb_x, tr_bb_y = theano_format(train_samples, batch_size)

    if argv.dev_data:
        dev_dataset, dev_bb_x, dev_bb_y = theano_format(dev_samples, batch_size)

    if argv.test_data:
        te_dataset, te_bb_x, te_bb_y = theano_format(test_samples, batch_size)

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    """ Set a model """
    print '\n\nBuilding a model...'
    model = set_model(argv=argv, emb=emb, vocab=vocab_word)
    train_f = set_train_f(model, tr_dataset)

    if argv.dev_data:
        dev_f = set_predict_f(model, dev_dataset)

    if argv.test_data:
        test_f = set_predict_f(model, te_dataset)

    ###############
    # TRAIN MODEL #
    ###############

    print '\nTRAINING START\n'
    indices = range(len(tr_bb_y))
    best_dev_acc = -1.
    best_test_acc = -1.

    for epoch in xrange(argv.epoch):
        print '\n\nEPOCH: %d' % (epoch + 1)
        print '\tTRAIN\n\t',

        np.random.shuffle(indices)
        start = time.time()

        ttl_nll = 0.
        ttl_crr = 0.
        for i, b_index in enumerate(indices):
            if (i + 1) % 100 == 0:
                print i + 1,
                sys.stdout.flush()

            bb_x_i = tr_bb_x[b_index]
            bb_y_i = tr_bb_y[b_index]
            crr, nll = train_f(index=b_index, bob_x=bb_x_i[0], eob_x=bb_x_i[1], bob_y=bb_y_i[0], eob_y=bb_y_i[1])
            ttl_crr += np.sum(crr)
            ttl_nll += nll

        end = time.time()
        print '\n\tTime: %f\tNLL: %f' % ((end - start), ttl_nll)
        print '\tACC: %f  CRR: %d   TOTAL: %d' % (ttl_crr/n_tr_samples, ttl_crr, n_tr_samples)

        update = False
        if argv.dev_data:
            print '\n\tDEV\n\t',
            dev_acc = predict(dev_f, dev_bb_x, dev_bb_y, n_dev_samples)
            if best_dev_acc < dev_acc:
                best_dev_acc = dev_acc
                update = True

        if argv.test_data:
            print '\n\tTEST\n\t',
            test_acc = predict(test_f, te_bb_x, te_bb_y, n_te_samples)
            if update:
                best_test_acc = test_acc

        print '\n\tBEST DEV ACC: %f  TEST ACC: %f' % (best_dev_acc, best_test_acc)


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

    return acc


def main(argv):
    train(argv)

