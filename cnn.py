from utils import sample_weights, build_shared_zeros, build_shared_pad, sigmoid, tanh, relu
from optimizer import ada_grad, ada_delta, adam, sgd

import theano
import theano.tensor as T


class Model(object):
    def __init__(self, x, y, l, window, opt, lr, init_emb, dim_emb, dim_hidden, n_vocab, L2_reg, unit, activation=tanh):
        self.tr_inputs = [x, y, l]
        self.pr_inputs = [x, y, l]

        self.x = x  # 1D: batch_size * l * 2, 2D: window; elem=word_id
        self.y = y  # 1D: batch_size; elem=label
        self.l = l  # scalar: elem=sentence length

        batch_size = y.shape[0]
        self.zero = T.zeros((1, 1, dim_emb * window), dtype=theano.config.floatX)

        self.pad = build_shared_zeros((1, dim_emb))
        if init_emb is None:
            self.emb = theano.shared(sample_weights(n_vocab - 1, dim_emb))
        else:
            self.emb = theano.shared(init_emb)
        self.E = T.concatenate([self.pad, self.emb], 0)
        self.W_in = theano.shared(sample_weights(dim_emb * window, dim_hidden))
        self.W_h = theano.shared(sample_weights(dim_hidden, dim_hidden / 2))
        self.W_out = theano.shared(sample_weights(dim_hidden / 2, dim_hidden / 2))
        self.params = [self.emb, self.W_in, self.W_out, self.W_h]

        e = self.E[x]  # e: 1D: batch_size * n_words * 2, 2D: window, 3D: dim_emb
        x_in = e.reshape((batch_size * 2, l, -1))

        z = T.max(self.zero_pad_filtering_gate(x_in, dim_emb, window) * relu(T.dot(x_in, self.W_in)), 1)
        h = activation(T.dot(z, self.W_h))
        h_1 = h[T.arange(batch_size) * 2]
        h_2 = h[T.arange(1, batch_size + 1) * 2 - 1]
        self.p_y = sigmoid(T.batched_dot(T.dot(h_1, self.W_out), h_2))

        """ Objective Function """
        self.nll = loss_function(self.p_y, y)
        self.L2_sqr = regularization(self.params)
        self.cost = self.nll + L2_reg * self.L2_sqr / 2.

        """ Optimization """
        if opt == 'adagrad':
            self.update = ada_grad(cost=self.cost, params=self.params, lr=lr)
        elif opt == 'ada_delta':
            self.update = ada_delta(cost=self.cost, params=self.params)
        elif opt == 'adam':
            self.update = adam(cost=self.cost, params=self.params)
        else:
            self.update = sgd(cost=self.cost, params=self.params, lr=lr)

        """ Predicts """
        self.y_hat = T.switch(self.p_y > 0.5, 1, 0)  # 1D: batch

        """ Check Accuracies """
        self.correct = T.eq(self.y_hat, y)

    def zero_pad_filtering_gate(self, matrix, dim_emb, window):
        return T.neq(T.sum(T.eq(matrix, self.zero), 2, keepdims=True), dim_emb * window)


def convolution(l_i, bos, x, W):
    eos = bos + l_i
    x_i = x[bos: eos]  # 1D: n_words, 2D: window * dim_emb
    h_i = T.max(T.dot(x_i, W), 0)
    return h_i, eos


def loss_function(p_y, y):
    return - T.sum(y * T.log(p_y) + (1 - y) * T.log(1. - p_y))


def regularization(params):
    return reduce(lambda a, b: a + T.sum(b ** 2), params, 0.)
