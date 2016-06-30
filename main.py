import numpy as np
import theano

theano.config.floatX = 'float32'
np.random.seed(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Question Answering System')

    parser.add_argument('-mode', default='xml', help='xml/train/test')
    parser.add_argument('-posts', help='path to posts')
    parser.add_argument('--task', default='binary', help='binary/retrieval')
    parser.add_argument('--links', help='path to links')
    parser.add_argument('--check', default=False, help='check')

    parser.add_argument('--p_data', help='path to data')
    parser.add_argument('--n_data', help='path to data')

    parser.add_argument('--train_data', help='path to data')

    """ Neural Architectures """
    parser.add_argument('--dim_emb',    type=int, default=50, help='dimension of embeddings')
    parser.add_argument('--dim_hidden', type=int, default=50, help='dimension of hidden layer')
    parser.add_argument('--unit', default='gru', help='unit')
    parser.add_argument('--window', type=int, default=5, help='window size for convolution')
    parser.add_argument('--activation', default='tanh', help='activation')

    """ Training Parameters """
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch size')
    parser.add_argument('--opt', default='sgd', help='optimization method')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0075, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--init_emb', default=None, help='Initial embedding to be loaded')

    """ Starting a mode """
    argv = parser.parse_args()
    if argv.mode == 'xml':
        import xml_parser
        xml_parser.main(argv)
    elif argv.mode == 'train':
        import train
        train.main(argv)
    elif argv.mode == 'sep':
        import separate_dataset
        separate_dataset.main(argv)
