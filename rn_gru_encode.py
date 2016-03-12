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

'''
    def _build_g(self):
        """Build Theano graph and define generation functions."""

        stdout.write("Compiling generation functions...")
        stdout.flush()
        time1 = time.time()

        # Local binding for convenience
        forward_step = self._forward_step

        ### SEQUENCE GENERATION ###

        x_in = T.vector('x_in')
        s_in = T.matrix('s_in')
        k = T.iscalar('k')
        temperature = T.scalar('temperature')

        rng = T.shared_randomstreams.RandomStreams(seed=int(
            np.sum(self.a.get_value()) * np.sum(self.b.get_value()) 
            * np.sum(self.c.get_value()) * 100000.0 + 123456789) % 4294967295)

        # Generate output sequence based on input single onehot and given state.
        # Chooses output char by multinomial, and feeds back in for next step.
        # Scaled by temperature parameter before softmax (temperature 1.0 leaves
        # softmax output unchanged).
        # Returns matrix of one-hot vectors
        def generate_step(x_t, s_t, temp):
            # Do next step
            o_t1, s_t = forward_step(x_t, s_t)

            # Get softmax
            o_t2 = T.nnet.softmax(o_t1 / temp)[-1]

            # Randomly choose by multinomial distribution
            o_rand = rng.multinomial(n=1, pvals=o_t2, dtype=th.config.floatX)

            return o_rand, s_t

        [o_chs, s_chs], genupdate = th.scan(
            fn=generate_step,
            outputs_info=[dict(initial=x_in), dict(initial=s_in)],
            non_sequences=temperature,
            n_steps=k)
        s_ch = s_chs[-1]

        self.gen_chars = th.function(
            inputs=[k, x_in, s_in, th.Param(temperature, default=0.1)], 
            outputs=[o_chs, s_ch], 
            name='gen_chars', 
            updates=genupdate)

        # Chooses output char by argmax, and feeds back in
        def generate_step_max(x_t, s_t):
            # Do next step
            o_t1, s_t1 = forward_step(x_t, s_t)

            # Get softmax
            o_t2 = T.nnet.softmax(o_t1)[-1]

            # Now find selected index
            o_idx = T.argmax(o_t2)

            # Create one-hot
            o_ret = T.zeros_like(o_t2)
            o_ret = T.set_subtensor(o_ret[o_idx], 1.0)

            return o_ret, s_t1

        [o_chms, s_chms], _ = th.scan(
            fn=generate_step_max,
            outputs_info=[dict(initial=x_in), dict(initial=s_in)],
            n_steps=k)
        s_chm = s_chms[-1]

        self.gen_chars_max = th.function(
            inputs=[k, x_in, s_in], 
            outputs=[o_chms, s_chm], 
            name='gen_chars_max')

        time2 = time.time()
        stdout.write("done!\nCompilation took {0:.3f} s.\n\n".format(time2 - time1))
        stdout.flush()
        self._built_g = True

    def _build_t(self):
        """Build Theano graph and define training functions."""

        stdout.write("Compiling training functions...")
        stdout.flush()
        time1 = time.time()

        # Local binding for convenience
        forward_step = self._forward_step

        ### SINGLE-SEQUENCE TRAINING ###

        # Inputs
        x = T.matrix('x')
        y = T.matrix('y')
        s_in = T.matrix('s_in')

        def single_step(x_t, s_t):
            o_t1, s_t = forward_step(x_t, s_t)
            # Theano's softmax returns matrix, and we just want the one entry
            o_t2 = T.nnet.softmax(o_t1)[-1]
            return o_t2, s_t

        # Now get Theano to do the heavy lifting
        [o, s_seq], _ = th.scan(
            single_step, 
            sequences=x, 
            truncate_gradient=self.hyper.bptt_truncate,
            outputs_info=[None, dict(initial=s_in)])
        s_out = s_seq[-1]

        # Costs
        o_errs = T.nnet.categorical_crossentropy(o, y)
        o_err = T.sum(o_errs)
        # Should regularize at some point
        cost = o_err

        # Gradients
        dparams = [ T.grad(cost, p) for p in self.params ]

        # rmsprop parameter updates
        learnrate = T.scalar('learnrate')
        decayrate = T.scalar('decayrate')

        uparams = [ decayrate * mp + (1 - decayrate) * dp ** 2 for mp, dp in zip(self.mparams, dparams) ]

        # Gather updates
        train_updates = OrderedDict()
        # Apply rmsprop updates to parameters
        for p, dp, up in zip(self.params, dparams, uparams):
            train_updates[p] = p - learnrate * dp / T.sqrt(up + 1e-6)
        # Update rmsprop caches
        for mp, up in zip(self.mparams, uparams):
            train_updates[mp] = up

        # Training step function
        self.train_step = th.function(
            inputs=[x, y, s_in, th.Param(learnrate, default=0.001), th.Param(decayrate, default=0.95)],
            outputs=s_out,
            updates=train_updates,
            name = 'train_step')

        ### BATCH-SEQUENCE TRAINING ###

        # Batch Inputs
        x_bat = T.tensor3('x_bat')
        y_bat = T.tensor3('y_bat')
        s_in_bat = T.tensor3('s_in_bat')

        # Costs

        # NEW VERSION
        # Since Theano's categorical cross-entropy function only works on matrices,
        # the cross-entropy loss has been moved inside the scan, so we can keep the
        # sequencing intact. (Slower, but seems to catch longer-term dependencies better?)
        def batch_step(x_t, y_t, s_t):
            o_t1, s_t = forward_step(x_t, s_t)
            # We can use the whole matrix from softmax for batches
            o_t2 = T.nnet.softmax(o_t1)
            # Get cross-entropy loss of batch step
            e_t = T.sum(T.nnet.categorical_crossentropy(o_t2, y_t))
            return e_t, s_t

        [o_errs_bat, s_seq_bat], _ = th.scan(
            batch_step, 
            sequences=[x_bat, y_bat], 
            truncate_gradient=self.hyper.bptt_truncate,
            outputs_info=[None, dict(initial=s_in_bat)])
        s_out_bat = s_seq_bat[-1]
        cost_bat = T.sum(o_errs_bat)

        # OLD VERSION
        # We have to reshape the outputs, since Theano's categorical cross-entropy
        # function will only work with matrices or vectors, not tensor3s.
        # Thus we flatten along the sequence/batch axes, leaving the prediction
        # vectors as-is, and this seems to be enough for Theano's deep magic to work.
        #o_bat_flat = T.reshape(o_bat, (o_bat.shape[0] * o_bat.shape[1], -1))
        #y_bat_flat = T.reshape(y_bat, (y_bat.shape[0] * y_bat.shape[1], -1))
        #o_errs_bat = T.nnet.categorical_crossentropy(o_bat_flat, y_bat_flat)
        #cost_bat = T.sum(o_errs_bat)

        # Gradients
        dparams_bat = [ T.grad(cost_bat, p) for p in self.params ]

        # rmsprop parameter updates
        uparams_bat = [ decayrate * mp + (1 - decayrate) * dp ** 2 for mp, dp in zip(self.mparams, dparams_bat) ]

        # Gather updates
        train_updates_bat = OrderedDict()
        # Apply rmsprop updates to parameters
        for p, dp, up in zip(self.params, dparams_bat, uparams_bat):
            train_updates_bat[p] = p - learnrate * dp / T.sqrt(up + 1e-6)
        # Update rmsprop caches
        for mp, up in zip(self.mparams, uparams_bat):
            train_updates_bat[mp] = up

        # Batch training step function
        self.train_step_bat = th.function(
            inputs=[x_bat, y_bat, s_in_bat, th.Param(learnrate, default=0.001), th.Param(decayrate, default=0.95)],
            outputs=s_out_bat,
            updates=train_updates_bat,
            name='train_step_bat')

        ### ERROR CHECKING ###

        # Error/cost calculations
        self.errs = th.function(
            inputs=[x, y, s_in], 
            outputs=[o_errs, s_out])
        self.errs_bat = th.function(
            inputs=[x_bat, y_bat, s_in_bat], 
            outputs=[o_errs_bat, s_out_bat])
        self.err = th.function(
            inputs=[x, y, s_in], 
            outputs=[cost, s_out])
        self.err_bat = th.function(
            inputs=[x_bat, y_bat, s_in_bat], 
            outputs=[cost_bat, s_out_bat])

        # Gradient calculations
        # We'll use this at some point for gradient checking
        self.grad = th.function(
            inputs=[x, y, s_in], 
            outputs=dparams)
        self.grad_bat = th.function(
            inputs=[x_bat, y_bat, s_in_bat], 
            outputs=dparams_bat)

        ### Whew, I think we're done! ###
        time2 = time.time()
        stdout.write("done!\nCompilation took {0:.3f} s.\n\n".format(time2 - time1))
        stdout.flush()
        self._built_t = True

    @classmethod
    def loadfromfile(cls, infile):
        with np.load(infile) as f:
            # Extract hyperparams and position
            p = f['p']
            hparams = pickle.loads(p.tobytes())
            hyper, epoch, pos = hparams['hyper'], hparams['epoch'], hparams['pos']

            # Load matrices
            pvalues = { n:f[n] for n in cls.pnames }

            # Create instance
            if isinstance(infile, str):
                stdout.write("Loaded model parameters from {0}\n".format(infile))
            stdout.write("Rebuilding model...\n")
            model = cls(hyper, epoch, pos, pvalues)

            return model

    def savetofile(self, outfile):
        # Pickle non-matrix params into bytestring, then convert to numpy byte array
        pklbytes = pickle.dumps({'hyper': self.hyper, 'epoch': self.epoch, 'pos': self.pos}, 
            protocol=pickle.HIGHEST_PROTOCOL)
        p = np.fromstring(pklbytes, dtype=np.uint8)

        # Gather parameter matrices and names
        pvalues = { n:m.get_value() for m, n in zip(self.params, self.pnames) }

        # Now save params and matrices to file
        try:
            np.savez(outfile, p=p, **pvalues)
        except OSError as e:
            raise e
        else:
            if isinstance(outfile, str):
                stdout.write("Saved model parameters to {0}\n".format(outfile))

    def freshstate(self, batchsize=0):
        if batchsize > 0:
            return np.zeros([self.hyper.layers, batchsize, self.hyper.state_size], dtype=th.config.floatX)
        else:
            return np.zeros([self.hyper.layers, self.hyper.state_size], dtype=th.config.floatX)
'''
