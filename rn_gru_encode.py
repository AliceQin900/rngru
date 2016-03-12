#import datetime, pickle, random, time
#from sys import stdin, stdout, stderr
import numpy as np
import theano as th
import theano.tensor as T

from rn_rnn_model import ModelParams

# TODO: add dropout between layers, *proper* embedding layer hooks
# TODO: make scaffolding for context windows (might need to involve charset)
class GRUEncode(ModelParams):
    """Multi-layer GRU network, with non-recurrent input layer E and 
    output layer V, with bias vectors of a and c, and specified number 
    of hidden layers.

    Each hidden layer has weight matrices U and W with bias vector b.

    Softmax applied to final output.
    """

    # Parameter and rmsprop cache matrix names
    pnames = ['E', 'a', 'U', 'W', 'b', 'V', 'c']
    mnames = ['mE', 'ma', 'mU', 'mW', 'mb', 'mV', 'mc']

    def __init__(self, hyper, epoch=0, pos=0, params=None):
        super(GRUEncode, self).__init__(hyper, epoch, pos)

        if not params:
            params = self._build_p()

        # Initialize shared variables

        # Parameter matrices
        self.E, self.a, self.U, self.W, self.b, self.V, self.c = [ 
            th.shared(name=p, value=params[p].astype(th.config.floatX)) 
            for p in self.pnames ]

        self.params = [self.E, self.a, self.U, self.W, self.b, self.V, self.c]

        # rmsprop parameters
        self.mE, self.ma, self.mU, self.mW, self.mb, self.mV, self.mc = [ 
            th.shared(name=m, value=np.zeros_like(params[p]).astype(th.config.floatX)) 
            for m, p in zip(self.mnames, self.pnames) ]

        self.mparams = [self.mE, self.ma, self.mU, self.mW, self.mb, self.mV, self.mc]

        # Build Theano generation functions
        self._build_g()

    def _build_p(self):
        '''Initialize parameter matrices.'''

        params = {}

        # Randomly initialize matrices if not provided
        # U, W get 3 2D matrices per layer (reset and update gates plus hidden state)
        params['E'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.vocab_size), np.sqrt(1.0/self.hyper.vocab_size), 
            (self.hyper.vocab_size, self.hyper.state_size))

        params['U'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            (self.hyper.layers*3, self.hyper.state_size, self.hyper.state_size))

        params['W'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            (self.hyper.layers*3, self.hyper.state_size, self.hyper.state_size))

        params['V'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            (self.hyper.state_size, self.hyper.vocab_size))

        # Initialize bias matrices to zeroes
        # b gets 3x2D per layer, c is single 2D
        params['a'] = np.zeros(self.hyper.state_size)
        params['b'] = np.zeros((self.hyper.layers*3, self.hyper.state_size))
        params['c'] = np.zeros(self.hyper.vocab_size)

        return params

    # Forward propagation
    def _forward_step(self, x_t, s_t):
        """Input vector/matrix x(t) and state matrix s(t)."""

        # Gradient clipping
        E, a, U, W, b, V, c = [th.gradient.grad_clip(p, -5.0, 5.0) for p in self.params]

        # Initialize state to return
        s_next = T.zeros_like(s_t)

        # Vocab-to-state encoding layer
        inout = T.tanh(T.dot(x_t, E) + a)

        # Loop over GRU layers
        for layer in range(self.hyper.layers):
            # 3 matrices per layer
            L = layer * 3
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

