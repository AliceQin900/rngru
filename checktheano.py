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
mat3 = theano.shared(name='mat3', value=matbase3.astype(theano.config.floatX))

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



rng = T.shared_randomstreams.RandomStreams(seed=6548619)
i_vec = T.vector('i_vec')

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

# one-hot testing
x_ints = T.ivector('x_ints')
x_dim = T.iscalar('x_dim')
to_onehot = theano.function([x_ints, x_dim], theano.tensor.extra_ops.to_one_hot(x_ints, x_dim))

x_mat = T.matrix('x_mat')
y_scale = T.scalar('scale')

y_onehot, updates = theano.scan(
    fn=lambda val, scale: scale * val,
    outputs_info=None,
    sequences=x_mat,
    non_sequences=y_scale)
scale_onehot = theano.function(inputs=[x_mat, y_scale], outputs=y_onehot)

def matdotvec(Xmat, Yvec):
    return T.dot(Xmat, Yvec.T)
y_dot, updates = theano.scan(
    fn=matdotvec,
    outputs_info=None,
    sequences=x,
    non_sequences=x_mat)
dot_onehot = theano.function(inputs=[x_mat, x], outputs=y_dot)

def dotcol(x, dim, mat_b):
    return mat_b.dot(T.eye(dim)[:,x])
y_col, updates = theano.scan(
    fn=dotcol,
    outputs_info=None,
    sequences=x_ints,
    non_sequences=[x_dim, x_mat])
col_onehot = theano.function(inputs=[x_ints, x_dim, x_mat], outputs=y_col)


# Run

testvecs = np.array([[0, 1], [1, 2], [2, 3]])
accinit = np.array([1, 1])
xes, acclast = dotandaddseq(testvecs, accinit)
print()
print(xes)
print()
print(acclast)
print("\n----\n")

accafter = dotandaddacc(acclast, 3)
print(accafter)
print("\n----\n")

accafterseq = dotandaddaccseq(acclast, 3)
print(accafterseq)
print("\n----\n")

accaftersoft = dotandsoftacc(acclast, 4)
print(accaftersoft)
print("\n----\n")

xv = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
xv_p, i_p = dotprob(xv, 5)
print(xv_p)
print()
print(i_p)
print("\n----\n")

xivec = np.array([1, 3, 2, 0, 4, 4, 2, 1], dtype='int32')
xidim = xivec[np.argmax(xivec)] + 1
xonehot = to_onehot(xivec, xidim)
print(xivec, xidim)
print()
print(xonehot)
print("\n----\n")

yonehot = scale_onehot(xonehot, 0.5)
print(yonehot)
print("\n----\n")

ydot = dot_onehot(xv_p, xonehot)
print(ydot)
print("\n----\n")

ycol = col_onehot(xivec, xidim, xv_p)
print(ycol)
print("\n----\n")


