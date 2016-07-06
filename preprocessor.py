import numpy as np
import theano

import io_utils

UNK = io_utils.UNK


def corpus_statistics(corpus):
    print '\nCORPUS STATISTICS'
    print 'Samples: %d' % len(corpus)


def sample_format(corpus, vocab, window=5):
    """
    :param corpus: 1D: n_samples; elem=(((title, body), (title, body)), label)
    :return:
    """

    phi_samples = []
    for sample in corpus:
        phi_sample = []
        for question in sample[0]:
            q = convert_into_ids(question[1], vocab)
            phi_sample.append(extract_features(q, window))
        phi_samples.append((phi_sample, sample[1]))
    return phi_samples


def convert_into_ids(sent, vocab):
    w_ids = []
    for w in sent:
        if w in vocab.w2i:
            w_ids.append(vocab.get_id(w))
        else:
            w_ids.append(vocab.get_id(UNK))
    return w_ids


def extract_features(sent, window=5):
    pad = [0 for i in xrange(window/2)]
    sample = pad + sent + pad
    return [sample[i: i+window] for i in xrange(len(sent))]


def theano_format(samples, batch_size):
    """
    :param samples: 1D: n_samples, 2D: n_words; elem=((feature, feature), label)
    :return:
    """

    def shared(_sample):
        return theano.shared(np.asarray(_sample, dtype='int32'), borrow=True)

    matrix = []
    labels = []
    q_length = []
    batch_boundary_for_x = []
    batch_boundary_for_y = []

    bob_x = 0
    bob_y = 0
    eob_y = 0
    max_length = -1
    tmp_matrix = []

    for i, sample in enumerate(samples):
        for q in sample[0]:
            tmp_matrix.append(q)
            if max_length < len(q):
                max_length = len(q)

        labels.append(sample[1])
        eob_y = i + 1

        if (i + 1) % batch_size == 0:
            batch_boundary_for_y.append((bob_y, eob_y))
            bob_y = eob_y

            matrix.extend(padding(tmp_matrix, max_length))
            q_length.append(max_length)
            batch_boundary_for_x.append((bob_x, len(matrix)))

            max_length = -1
            tmp_matrix = []
            bob_x = len(matrix)

    if len(tmp_matrix) > 0:
        batch_boundary_for_y.append((bob_y, eob_y))
        matrix.extend(padding(tmp_matrix, max_length))
        q_length.append(max_length)
        batch_boundary_for_x.append((bob_x, len(matrix)))

    return [shared(matrix), shared(labels), shared(q_length)], batch_boundary_for_x, batch_boundary_for_y


def padding(matrix, max_length, window=5):
    padded_matrix = []
    zero = [0 for j in xrange(window)]
    for sent in matrix:
        diff = max_length - len(sent)
        m = sent + [zero for i in xrange(diff)]
        padded_matrix.extend(m)

    assert len(padded_matrix) % max_length == 0
    return padded_matrix

