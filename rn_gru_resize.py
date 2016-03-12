#import datetime, pickle, random, time
#from sys import stdin, stdout, stderr
import numpy as np
import theano as th
import theano.tensor as T

from rn_rnn_model import ModelParams

# TODO: add dropout between layers, *proper* embedding layer hooks
# TODO: make scaffolding for context windows (might need to involve charset)
class GRUResize(ModelParams):
    """Multi-layer GRU network, but first layer has vocab-sized vector input and
    state-sized vector output, while additional layers input and output state-sized
    vectors.
    Matrices E and F are first-layer equivalents to U and W in subsequent layers, 
    renamed for clarity.
    Matrix V transforms state-sized output of last layer to vocab-sized vector, 
    and softmax applied to final output.
    """

    # Parameter and rmsprop cache matrix names
    pnames = ['E', 'F', 'a', 'U', 'W', 'b', 'V', 'c']
    mnames = ['mE', 'mF', 'ma', 'mU', 'mW', 'mb', 'mV', 'mc']

    def __init__(self, hyper, epoch=0, pos=0, params=None):
        super(GRUResize, self).__init__(hyper, epoch, pos)

        if not params:
            params = self._build_p()

        # Initialize shared variables

        # Parameter matrices
        self.E, self.F, self.a, self.U, self.W, self.b, self.V, self.c = [
            th.shared(name=p, value=params[p].astype(th.config.floatX)) 
            for p in self.pnames ]

        self.params = [self.E, self.F, self.a, self.U, self.W, self.b, self.V, self.c]

        # rmsprop parameters
        self.mE, self.mF, self.ma, self.mU, self.mW, self.mb, self.mV, self.mc = [ 
            th.shared(name=m, value=np.zeros_like(params[p]).astype(th.config.floatX)) 
            for m, p in zip(self.mnames, self.pnames) ]

        self.mparams = [self.mE, self.mF, self.ma, self.mU, self.mW, self.mb, self.mV, self.mc]

        # Build Theano generation functions
        self._build_g()

    def _build_p(self):
        '''Initialize parameter matrices.'''

        params = {}

        # Randomly initialize matrices if not provided
        # E, F, U, W get 3 2D matrices per layer (reset and update gates plus hidden state)
        params['E'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.vocab_size), np.sqrt(1.0/self.hyper.vocab_size), 
            (3, self.hyper.vocab_size, self.hyper.state_size))

        params['F'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            (3, self.hyper.state_size, self.hyper.state_size))

        params['U'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            ((self.hyper.layers-1)*3, self.hyper.state_size, self.hyper.state_size))

        params['W'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            ((self.hyper.layers-1)*3, self.hyper.state_size, self.hyper.state_size))

        params['V'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            (self.hyper.state_size, self.hyper.vocab_size))

        # Initialize bias matrices to zeroes
        # a and b are 2D, c is 1D
        params['a'] = np.zeros((3, self.hyper.state_size))
        params['b'] = np.zeros(((self.hyper.layers-1)*3, self.hyper.state_size))
        params['c'] = np.zeros(self.hyper.vocab_size)

        return params

    # Forward propagation
    def _forward_step(self, x_t, s_t):
        """Input vector/matrix x(t) and state matrix s(t)."""

        # Gradient clipping
        E, F, a, U, W, b, V, c = [th.gradient.grad_clip(p, -5.0, 5.0) for p in self.params]

        # Initialize state to return
        s_next = T.zeros_like(s_t)

        # Input GRU layer
        # Get previous state for this layer
        s_prev = s_t[0]
        # Update gate
        z = T.nnet.hard_sigmoid(T.dot(x_t, E[0]) + T.dot(s_prev, F[0]) + a[0])
        # Reset gate
        r = T.nnet.hard_sigmoid(T.dot(x_t, E[1]) + T.dot(s_prev, F[1]) + a[1])
        # Candidate state
        h = T.tanh(T.dot(x_t, E[2]) + T.dot(r * s_prev, F[2]) + a[2])
        # New state
        s_new = (T.ones_like(z) - z) * h + z * s_prev
        s_next = T.set_subtensor(s_next[0], s_new)
        # Update for next layer (might add dropout here later)
        inout = s_new

        # Loop over subsequent GRU layers
        for layer in range(1, self.hyper.layers):
            # 3 matrices per layer
            L = (layer-1) * 3
            # Get previous state for this layer
            s_prev = s_t[layer]
            # Update gate
            z = T.nnet.hard_sigmoid(T.dot(inout, U[L]) + T.dot(s_prev, W[L]) + b[L])
            # Reset gate
            r = T.nnet.hard_sigmoid(T.dot(inout, U[L+1]) + T.dot(s_prev, W[L+1]) + b[L+1])
            # Candidate state
            h = T.tanh(T.dot(inout, U[L+2]) + T.dot(r * s_prev, W[L+2]) + b[L+2])
            # New state
            s_new = (T.ones_like(z) - z) * h + z * s_prev
            s_next = T.set_subtensor(s_next[layer], s_new)
            # Update for next layer or final output (might add dropout here later)
            inout = s_new

        # Final output
        o_t = T.dot(inout, V) + c
        return o_t, s_next


