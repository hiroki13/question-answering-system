import numpy as np
import theano

theano.config.floatX = 'float32'
np.random.seed(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Question Answering System')

    parser.add_argument('-mode', default='train', help='data/train/test')
    parser.add_argument('-posts', help='path to posts')
    parser.add_argument('--task', default='binary', help='data/binary/ranking/retrieval')
    parser.add_argument('--data_type', default='positive', help='positive/negative')
    parser.add_argument('--links', help='path to links')
    parser.add_argument('--check', default=False, help='check')
    parser.add_argument('--data_size', type=int, default=1000000, help='data size')
    parser.add_argument('--n_cands', type=int, default=2, help='num of negative candidates')

    parser.add_argument('--p_data', help='path to data')
    parser.add_argument('--n_data', help='path to data')

    parser.add_argument('--train_data', default=None, help='path to data')
    parser.add_argument('--dev_data', default=None, help='path to data')
    parser.add_argument('--test_data', default=None, help='path to data')

    """ Neural Architectures """
    parser.add_argument('--dim_emb',    type=int, default=50, help='dimension of embeddings')
    parser.add_argument('--dim_hidden', type=int, default=50, help='dimension of hidden layer')
    parser.add_argument('--unit', default='gru', help='unit')
    parser.add_argument('--window', type=int, default=5, help='window size for convolution')
    parser.add_argument('--layer',  type=int, default=1,         help='number of layers')
    parser.add_argument('--sim',  default='linear',         help='number of layers')
    parser.add_argument('--activation', default='tanh', help='activation')

    """ Training Parameters """
    parser.add_argument('--save', type=bool, default=False, help='save model files')
    parser.add_argument('--load_vocab', type=str, default=None, help='save model files')
    parser.add_argument('--load_model', type=str, default=None, help='save model files')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch size')
    parser.add_argument('--opt', default='adam', help='optimization method')
    parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--reg', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--init_emb', default=None, help='Initial embedding to be loaded')

    """ Starting a mode """
    argv = parser.parse_args()
    if argv.mode == 'data':
        if argv.task == 'data':
            import data_generator
            data_generator.main(argv)
        elif argv.task == 'binary':
            import binary_dataset_generator
            binary_dataset_generator.main(argv)
        elif argv.task == 'ranking':
            import ranking_dataset_generator
            ranking_dataset_generator.main(argv)
        elif argv.task == 'match':
            import similar_question_generator
            similar_question_generator.main(argv)
        elif argv.task == 'tokenize':
            import tokenizer
            tokenizer.main(argv)
    elif argv.mode == 'train':
        import train
        train.main(argv)
    elif argv.mode == 'test':
        import test
        test.main(argv)
    elif argv.mode == 'tfidf':
        import tfidf
        tfidf.main(argv)
