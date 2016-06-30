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

    samples = []
    for sample in corpus:
        q1 = convert_into_ids(sample[0][0][1], vocab)
        q2 = convert_into_ids(sample[0][1][1], vocab)

        q1_phi = extract_features(q1, window)
        q2_phi = extract_features(q2, window)
        samples.append(([q1_phi, q2_phi], sample[1]))
    return samples


def convert_into_ids(sent, vocab):
    w_ids = []
    for w in sent:
        if w in vocab.w2i:
            w_ids.append(vocab.get_id(w))
        else:
            w_ids.append(vocab[UNK])
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
        q1 = sample[0][0]
        q2 = sample[0][1]
        tmp_matrix.append(q1)
        tmp_matrix.append(q2)

        labels.append(sample[1])

        q1_len = len(q1)
        q2_len = len(q2)

        eob_y = i + 1

        if max_length < q1_len:
            max_length = q1_len
        if max_length < q2_len:
            max_length = q2_len

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


def separate_datasets(samples):
    n_samples = len(samples)
    sep = n_samples / 10
    train_data = samples[: sep * 8]
    dev_data = samples[sep * 8: sep * 8 + sep / 2]
    test_data = samples[sep * 8 + sep / 2:]

    print 'TRAIN DATA: %d\tDEV DATA: %d\tTEST DATA: %d' % (len(train_data), len(dev_data), len(test_data))
    return train_data, dev_data, test_data


