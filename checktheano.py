#!/usr/bin/env python3

import numpy as np
import theano
import theano.tensor as T
theano.config.exception_verbosity='high'

matbase = np.array([[1.0, 0.0], [0.0, 1.0]])
mat = theano.shared(name='mat', value=matbase.astype(theano.config.floatX))
matbase2 = np.array([[1.0, 0.5], [0.5, 1.0]])
mat2 = theano.shared(name='mat2', value=matbase2.astype(theano.config.floatX))
matbase3 = np.identity(5)
mat3 = theano.shared(name='mat2', value=matbase3.astype(theano.config.floatX))

x = T.matrix('x')
acc = T.vector('acc')
upto = T.iscalar('upto')

def dotandadd(x_t, acc_t):
    x_new = mat.dot(x_t)
    a_new = acc_t + x_new
    return x_new, a_new

[x_seq, acc_seq], updates = theano.scan(
    fn=dotandadd,
    outputs_info=[None, dict(initial=acc)],
    sequences=[x])

acc_end = acc_seq[-1]
dotandaddseq = theano.function([x, acc], [x_seq, acc_end], allow_input_downcast=True)

k = T.iscalar('k')

def dotandadd2(acc_t):
    return mat.dot(acc_t) + acc_t

acc_k_seq, updates = theano.scan(
    fn=dotandadd2,
    outputs_info=[dict(initial=acc)],
    n_steps=k)

acc_k=acc_k_seq[-1]
dotandaddacc = theano.function([acc, k], acc_k, allow_input_downcast=True)
dotandaddaccseq = theano.function([acc, k], acc_k_seq, allow_input_downcast=True)

def dotandsoftmax(acc_t):
    return T.nnet.softmax(mat.dot(acc_t))[0]

acc_s_seq, updates = theano.scan(
    fn=dotandsoftmax,
    outputs_info=[dict(initial=acc)],
    n_steps=k)
acc_s = acc_s_seq[-1]
dotandsoftacc = theano.function([acc, k], acc_s, allow_input_downcast=True)

rng = T.shared_randomstreams.RandomStreams(seed=6547619)
i_vec = T.vector('i_vec')

#def pickprobof(acc_t, i_t):
#    o_prob = T.nnet.softmax(acc_t)[0]
#    i_next = rng.choice(size=(), a=2, p=o_prob).astype('int32')
#    return o_prob, i_next

#[acc_p_seq, i_seq], updates = theano.scan(
#    fn=pickprobof,
#    outputs_info=[dict(initial=acc), dict(initial=i_init)],
#    n_steps=k)
#pickaccsoft = theano.function([acc, i_init, k], acc_p_seq)

def dotandprob(x_vec):
    randsize = T.as_tensor_variable(np.asarray(0, dtype='int64'))
    randidx = rng.choice(size=1, a=T.arange(x_vec.shape[0]), p=x_vec, ndim=1)
    return mat3.dot(x_vec), randidx[0]

x_v = T.vector('x_v')
[x_v_seq, idx_seq], updates = theano.scan(
    fn=dotandprob,
    outputs_info=[dict(initial=x_v), None],
    n_steps=k)
dotprob = theano.function([x_v, k], [x_v_seq, idx_seq], updates=updates, allow_input_downcast=True)


# Run

testvecs = np.array([[0, 1], [1, 2], [2, 3]])
accinit = np.array([1, 1])
xv = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

xes, acclast = dotandaddseq(testvecs, accinit)
print(xes)
print()
print(acclast)
print()

accafter = dotandaddacc(acclast, 3)
print(accafter)
print()

accafterseq = dotandaddaccseq(acclast, 3)
print(accafterseq)
print()

accaftersoft = dotandsoftacc(acclast, 4)
print(accaftersoft)
print()

xv_p, i_p = dotprob(xv, 5)
print(xv_p)
print()
print(i_p)
print()


