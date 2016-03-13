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

    def __init__(self, hyper, epoch=0, pos=0, pvalues=None):
        super(GRUResize, self).__init__(hyper, epoch, pos, pvalues)

    def _build_p(self):
        '''Initialize parameter matrices.'''

        pvalues = {}

        # Randomly initialize matrices if not provided
        # E, F, U, W get 3 2D matrices per layer (reset and update gates plus hidden state)
        pvalues['E'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.vocab_size), np.sqrt(1.0/self.hyper.vocab_size), 
            (3, self.hyper.vocab_size, self.hyper.state_size))

        pvalues['F'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            (3, self.hyper.state_size, self.hyper.state_size))

        pvalues['U'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            ((self.hyper.layers-1)*3, self.hyper.state_size, self.hyper.state_size))

        pvalues['W'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            ((self.hyper.layers-1)*3, self.hyper.state_size, self.hyper.state_size))

        pvalues['V'] = np.random.uniform(
            -np.sqrt(1.0/self.hyper.state_size), np.sqrt(1.0/self.hyper.state_size), 
            (self.hyper.state_size, self.hyper.vocab_size))

        # Initialize bias matrices to zeroes
        # a and b are 2D, c is 1D
        pvalues['a'] = np.zeros((3, self.hyper.state_size))
        pvalues['b'] = np.zeros(((self.hyper.layers-1)*3, self.hyper.state_size))
        pvalues['c'] = np.zeros(self.hyper.vocab_size)

        return pvalues

    # Forward propagation
    def _forward_step(self, x_t, s_t):
        """Input vector/matrix x(t) and state matrix s(t)."""

        # Gradient clipping
        E, F, a, U, W, b, V, c = [ th.gradient.grad_clip(p, -5.0, 5.0) for p in self.params.values() ]

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

    # Regularization cost
    def _reg_cost(self, reg_lambda):
        weightsum = T.sum(T.sqr(self.params['E'])) + \
            T.sum(T.sqr(self.params['F'])) + \
            T.sum(T.sqr(self.params['U'])) + \
            T.sum(T.sqr(self.params['W'])) + \
            T.sum(T.sqr(self.params['V']))
        return reg_lambda * weightsum / 2.0

