import theano
import theano.tensor as T

import cnn
from utils import relu, tanh


def set_model(argv, emb, vocab):
    x = T.imatrix('x')
    y = T.ivector('y')
    l = T.iscalar('l')

    """ Set the classifier parameters"""
    window = argv.window
    opt = argv.opt
    lr = argv.lr
    init_emb = emb
    dim_emb = argv.dim_emb if emb is None else len(emb[0])
    dim_hidden = argv.dim_hidden
    n_vocab = vocab.size()
    L2_reg = argv.reg
    unit = argv.unit
    activation = relu if argv.activation == 'relu' else tanh

    model = cnn.Model(x=x, y=y, l=l, window=window, opt=opt, lr=lr, init_emb=init_emb, dim_emb=dim_emb,
                      dim_hidden=dim_hidden, n_vocab=n_vocab, L2_reg=L2_reg, unit=unit, activation=activation)
    return model


def set_train_f(model, tr_dataset):
    # dataset = [tr_x, tr_y, tr_l]
    # tr_x=features: 1D: n_samples * n_words, 2D: window; elem=word id
    # tr_y=labels: 1D: n_samples; elem=scalar
    # tr_l=question length: 1D: n_samples * 2; elem=scalar
    # bb_x=batch indices for x: 1D: n_samples / batch_size + 1; elem=(bob, eob)
    # bb_y=batch indices for y: 1D: n_samples / batch_size + 1; elem=(bob, eob)

    index = T.iscalar('index')
    bob_x = T.iscalar('bob_x')
    eob_x = T.iscalar('eob_x')
    bob_y = T.iscalar('bob_y')
    eob_y = T.iscalar('eob_y')

    train_f = theano.function(inputs=[index, bob_x, eob_x, bob_y, eob_y],
                              outputs=[model.correct, model.nll],
                              updates=model.update,
                              givens={
                                  model.tr_inputs[0]: tr_dataset[0][bob_x: eob_x],
                                  model.tr_inputs[1]: tr_dataset[1][bob_y: eob_y],
                                  model.tr_inputs[2]: tr_dataset[2][index],
                              }
                              )
    return train_f


def set_predict_f(model, te_dataset):
    # dataset = [tr_x, tr_y, tr_l]
    # tr_x=features: 1D: n_samples * n_words, 2D: window; elem=word id
    # tr_y=labels: 1D: n_samples; elem=scalar
    # tr_l=question length: 1D: n_samples * 2; elem=scalar
    # bb_x=batch indices for x: 1D: n_samples / batch_size + 1; elem=(bob, eob)
    # bb_y=batch indices for y: 1D: n_samples / batch_size + 1; elem=(bob, eob)

    index = T.iscalar('index')
    bob_x = T.iscalar('bob_x')
    eob_x = T.iscalar('eob_x')
    bob_y = T.iscalar('bob_y')
    eob_y = T.iscalar('eob_y')

    predict_f = theano.function(inputs=[index, bob_x, eob_x, bob_y, eob_y],
                                outputs=model.correct,
                                givens={
                                model.pr_inputs[0]: te_dataset[0][bob_x: eob_x],
                                model.pr_inputs[1]: te_dataset[1][bob_y: eob_y],
                                model.pr_inputs[2]: te_dataset[2][index],
                                }
                                )
    return predict_f
