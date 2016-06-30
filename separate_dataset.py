from io_utils import load_sep, save_sep
import random


def separate_data(argv):
    print '\nSEPARATE DATA\n'
    positive_data = load_sep(argv.p_data, 1)
    negative_data = load_sep(argv.n_data, 0)
    p_len = len(positive_data)
    n_len = len(negative_data)

    if p_len < n_len:
        dataset = positive_data + negative_data[:p_len]
    else:
        dataset = positive_data[:n_len] + negative_data

    print 'POSITIVE: %d\tNEGATIVE: %d\tTOTAL: %d' % (p_len, n_len, len(dataset))

    indices = range(len(dataset))
    random.shuffle(indices)

    randomized_dataset = [dataset[i] for i in indices]
    save_sep('dataset', randomized_dataset)


def main(argv):
    separate_data(argv)
