import numpy as np
import theano as th
import theano.tensor as T

from rn_rnn_model import ModelParams

# TODO: add dropout between layers, *proper* embedding layer hooks
# TODO: make scaffolding for context windows (might need to involve charset)
class GRUEncode(ModelParams):
    """Multi-layer GRU network, with non-recurrent input layer E and 
    output layer V, with bias vectors of a and c, and specified number 
    of gated recurrent hidden layers.

    Each hidden layer has weight matrices U and W with bias vector b.

    Softmax applied to final output.
    """

    # Parameter and rmsprop cache matrix names
    pnames = ['E', 'a', 'U', 'W', 'b', 'V', 'c']

    def __init__(self, hyper, epoch=0, pos=0, pvalues=None):
        super(GRUEncode, self).__init__(hyper, epoch, pos, pvalues)

    def _build_p(self):
        '''Initialize parameter matrices.'''

        pvalues = {}

        # Randomly initialize matrices if not provided
        # U, W get 3 2D matrices per layer (reset and update gates plus hidden state)
        pvalues['E'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.vocab_size), np.sqrt(1.0/self.hyper.vocab_size), 
            (self.hyper.vocab_size, self.hyper.state_size))

        pvalues['U'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            (self.hyper.layers*3, self.hyper.state_size, self.hyper.state_size))

        pvalues['W'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            (self.hyper.layers*3, self.hyper.state_size, self.hyper.state_size))

        pvalues['V'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            (self.hyper.state_size, self.hyper.vocab_size))

        # Initialize bias matrices to zeroes
        # b gets 3xLayers matrix, a and c are 1D vectors
        pvalues['a'] = np.zeros(self.hyper.state_size)
        pvalues['b'] = np.zeros((self.hyper.layers*3, self.hyper.state_size))
        pvalues['c'] = np.zeros(self.hyper.vocab_size)

        return pvalues

    # Forward propagation
    def _forward_step(self, x_t, s_t):
        """Input vector/matrix x(t) and state matrix s(t)."""

        # Gradient clipping
        E, a, U, W, b, V, c = [ th.gradient.grad_clip(p, -5.0, 5.0) for p in self.params.values() ]

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

    # Regularization cost
    def _reg_cost(self, reg_lambda):
        weightsum = T.sum(T.sqr(self.params['E'])) + \
            T.sum(T.sqr(self.params['U'])) + \
            T.sum(T.sqr(self.params['W'])) + \
            T.sum(T.sqr(self.params['V']))
        return reg_lambda * weightsum / 2.0

    # State matrix
    def freshstate(self, batchsize):
        if batchsize > 0:
            return np.zeros([self.hyper.layers, batchsize, self.hyper.state_size], dtype=th.config.floatX)
        else:
            return np.zeros([self.hyper.layers, self.hyper.state_size], dtype=th.config.floatX)

