import random
import gzip


def load(path, label):
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


def save(fn, data):
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


def separate_datasets(samples):
    n_samples = len(samples)
    sep = n_samples / 10
    train_data = samples[: sep * 8]
    dev_data = samples[sep * 8: sep * 8 + sep / 2]
    test_data = samples[sep * 8 + sep / 2:]

    print 'TRAIN DATA: %d\tDEV DATA: %d\tTEST DATA: %d' % (len(train_data), len(dev_data), len(test_data))
    return train_data, dev_data, test_data


def create_datasets(argv):
    print '\nSEPARATE DATA\n'
    positive_data = load(argv.p_data, 1)  # 1D: n_samples; 2D: 2, elem=(line, label)
    negative_data = load(argv.n_data, 0)  # 1D: n_samples; 2D: n_cands, elem=(line, label)
    p_len = len(positive_data)
    n_len = len(negative_data)

    random.shuffle(positive_data)
    random.shuffle(negative_data)

    if p_len < n_len:
        dataset = positive_data + negative_data[:p_len]
    else:
        dataset = positive_data[:n_len] + negative_data

    print 'POSITIVE: %d\tNEGATIVE: %d\tTOTAL: %d' % (p_len, n_len, len(dataset))

    indices = range(len(dataset))
    random.shuffle(indices)
    train_dataset, dev_dataset, test_dataset = separate_datasets([dataset[i] for i in indices])

    save('train.bi', train_dataset)
    save('dev.bi', dev_dataset)
    save('test.bi', test_dataset)


def main(argv):
    create_datasets(argv)
