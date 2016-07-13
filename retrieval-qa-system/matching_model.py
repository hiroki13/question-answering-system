from utils import sample_weights, build_shared_zeros, sigmoid, tanh, relu
from optimizer import ada_grad, ada_delta, adam, sgd
import cnn

import theano
import theano.tensor as T


class Model(object):
    def __init__(self, x, y, l, window, opt, lr, init_emb, dim_emb, dim_hidden, n_vocab, L2_reg, unit,
                 sim='cos', n_layers=1, activation=tanh):
        self.tr_inputs = [x, y, l]
        self.pr_inputs = [x, y, l]

        self.x = x  # 1D: batch_size * l * 2, 2D: window; elem=word_id
        self.y = y  # 1D: batch_size; elem=label
        self.l = l  # scalar: elem=sentence length

        batch_size = y.shape[0]
        n_cands = x.shape[0] / batch_size / l

        self.pad = build_shared_zeros((1, dim_emb))
        if init_emb is None:
            self.emb = theano.shared(sample_weights(n_vocab - 1, dim_emb))
        else:
            self.emb = theano.shared(init_emb)
        self.E = T.concatenate([self.pad, self.emb], 0)
        self.W_out = theano.shared(sample_weights(dim_hidden, dim_hidden))
        self.params = [self.emb, self.W_out]

        """ Input Layer """
        e = self.E[x]  # e: 1D: batch_size * l * 2, 2D: window, 3D: dim_emb
        x_in = e.reshape((batch_size * n_cands, l, -1))

        """ Intermediate Layer """
        # h: 1D: n_batch * n_cands, 2D: dim_emb
        h, params = cnn.layers(x_in, window, dim_emb, dim_hidden, n_layers, activation)
        self.params.extend(params)

        """ Output Layer """
        h_1 = h[T.arange(batch_size) * 2]
        h_2 = h[T.arange(1, batch_size + 1) * 2 - 1]
        if sim == 'cos':
            y_score = cosign_similarity(h_1, h_2)
        else:
            y_score = sigmoid(T.batched_dot(T.dot(h_1, self.W_out), h_2))

        """ Objective Function """
        if sim == 'cos':
            self.nll = mean_squared_loss(y_score, y)
        else:
            self.nll = binary_crass_entropy(y_score, y)
        self.L2_sqr = regularization(self.params)
        self.cost = self.nll + L2_reg * self.L2_sqr / 2.

        """ Optimization """
        if opt == 'adagrad':
            self.update = ada_grad(cost=self.cost, params=self.params, lr=lr)
        elif opt == 'ada_delta':
            self.update = ada_delta(cost=self.cost, params=self.params)
        elif opt == 'adam':
            self.update = adam(cost=self.cost, params=self.params, lr=lr)
        else:
            self.update = sgd(cost=self.cost, params=self.params, lr=lr)

        """ Predicts """
        y_binary = T.switch(y_score > 0.5, 1, 0)  # 1D: batch

        """ Check Accuracies """
        self.correct = T.eq(y_binary, y)


def cosign_similarity(x, y):
    return T.batched_dot(x, y.dimshuffle(0, 2, 1)) / (T.sum(x ** 2, 1, keepdims=True) * T.sum(y ** 2, 2))


def binary_crass_entropy(p_y, y):
    return - T.sum(y * T.log(p_y) + (1 - y) * T.log(1. - p_y))


def mean_squared_loss(p_y, y):
    return T.sum((y - p_y) ** 2 / 2)


def regularization(params):
    return reduce(lambda a, b: a + T.sum(b ** 2), params, 0.)
