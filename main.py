import numpy as np
import theano

theano.config.floatX = 'float32'
np.random.seed(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Question Answering System')

    parser.add_argument('-mode', default='xml', help='xml/train/test')
    parser.add_argument('-posts', help='path to posts')
    parser.add_argument('--links', help='path to links')
    parser.add_argument('--check', default=False, help='check')

    """ Starting a mode """
    argv = parser.parse_args()
    if argv.mode == 'xml':
        import xml_parser
        xml_parser.main(argv)
