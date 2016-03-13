# Python module dependencies
import pickle, time
from sys import stdin, stdout, stderr
from collections import OrderedDict
import numpy as np
import theano as th
import theano.tensor as T


class ModelParams:
    """Base class for RNN variants.
    NOTE: Not intended to be instantiated!
    """
    # Parameter matrix names and ordering
    # Defined by model subclass
    pnames = []

    def __init__(self, hyper, epoch=0, pos=0, pvalues=None):
        self.hyper = hyper
        self.epoch = epoch
        self.pos = pos

        if not pvalues:
            pvalues = self._build_p()

        # Initialize shared variables

        # Create parameter dicts
        # OrderedDict used to keep paramater access deterministic throughout
        self.params = OrderedDict()
        self.mparams = OrderedDict()

        # Load parameter matrices and create rmsprop caches
        for n in self.pnames:
            self.params[n] = th.shared(name=n, value=pvalues[n].astype(th.config.floatX))
            self.mparams['m'+n] = th.shared(name='m'+n, value=np.zeros_like(pvalues[n]).astype(th.config.floatX))

        # Build Theano generation functions
        self._built_g = False
        self._built_t = False
        self._build_g()

    # Model-specific definitions of parameters and forward propagation step function
    def _build_p(self):
        pass
    def _forward_step(self, x_t, s_t):
        pass
    def _weight_cost(self, reg_lambda):
        pass

    # Theano-generated model-dependent functions
    def gen_chars(self, *args, **kwargs):
        pass
    def gen_chars_max(self, *args, **kwargs):
        pass
    def train_step_bat(self, *args, **kwargs):
        pass
    def errs_bat(self, *args, **kwargs):
        pass
    def err_bat(self, *args, **kwargs):
        pass
    def grad_bat(self, *args, **kwargs):
        pass

    # Cross-model definitions of generation functions
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

        rng = T.shared_randomstreams.RandomStreams(seed=614731559)

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

    # Cross-model definitions of training functions
    def _build_t(self):
        """Build Theano graph and define training functions."""

        stdout.write("Compiling training functions...")
        stdout.flush()
        time1 = time.time()

        # Local bindings for convenience
        forward_step = self._forward_step
        weight_cost = self._weight_cost

        # Scalar training parameters
        learnrate = T.scalar('learnrate')
        decayrate = T.scalar('decayrate')
        reg_lambda = T.scalar('reg_lambda')

        ### BATCH-SEQUENCE TRAINING ###

        # Batch Inputs
        x_bat = T.tensor3('x_bat')
        y_bat = T.tensor3('y_bat')
        s_in_bat = T.tensor3('s_in_bat')

        # Costs

        def batch_step(x_t, y_t, s_t):
            o_t1, s_t = forward_step(x_t, s_t)
            # We can use the whole matrix from softmax for batches
            o_t2 = T.nnet.softmax(o_t1)
            return o_t2, s_t

        [o_bat, s_seq_bat], _ = th.scan(
            batch_step, 
            sequences=[x_bat, y_bat], 
            truncate_gradient=self.hyper.bptt_truncate,
            outputs_info=[None, dict(initial=s_in_bat)])
        s_out_bat = s_seq_bat[-1]

        # We have to reshape the outputs, since Theano's categorical cross-entropy
        # function will only work with matrices or vectors, not tensor3s.
        # Thus we flatten along the sequence/batch axes, leaving the prediction
        # vectors as-is, compute cross-entropy, then reshape the errors back to 
        # their proper dimensions.
        o_bat_flat = T.reshape(o_bat, (o_bat.shape[0] * o_bat.shape[1], -1))
        y_bat_flat = T.reshape(y_bat, (y_bat.shape[0] * y_bat.shape[1], -1))
        o_errs_bat = T.nnet.categorical_crossentropy(o_bat_flat, y_bat_flat)
        o_errs_res = T.reshape(o_errs_bat, (o_bat.shape[0], o_bat.shape[1]))

        # Next, we reshuffle to group sequences together instead
        # of batches, then sum the individual sequence errors
        # (Hopefully Theano's auto-differentials follow this)
        o_errs_shuf = o_errs_res.dimshuffle(1, 0)
        o_errs_sums = T.sum(o_errs_shuf, axis=1)
        # Final cost (with regularization)
        # (weight_cost() defined per-model)
        cost_bat = T.sum(o_errs_sums) + weight_cost(reg_lambda)

        # Gradients
        dparams_bat = [ T.grad(cost_bat, p) for p in self.params.values() ]

        # rmsprop parameter updates
        uparams_bat = [ decayrate * mp + (1 - decayrate) * dp ** 2 for mp, dp in zip(self.mparams.values(), dparams_bat) ]

        # Gather updates
        train_updates_bat = OrderedDict()
        # Apply rmsprop updates to parameters
        for p, dp, up in zip(self.params.values(), dparams_bat, uparams_bat):
            train_updates_bat[p] = p - learnrate * dp / T.sqrt(up + 1e-6)
        # Update rmsprop caches
        for mp, up in zip(self.mparams.values(), uparams_bat):
            train_updates_bat[mp] = up

        # Batch training step function
        self.train_step_bat = th.function(
            inputs=[x_bat, y_bat, s_in_bat, 
                th.Param(learnrate, default=0.001), 
                th.Param(decayrate, default=0.95),
                th.Param(reg_lambda, default=0.1)],
            outputs=s_out_bat,
            updates=train_updates_bat,
            name='train_step_bat')

        ### ERROR CHECKING ###

        # Error/cost calculations
        self.errs_bat = th.function(
            inputs=[x_bat, y_bat, s_in_bat], 
            outputs=[o_errs_res, s_out_bat])
        self.err_bat = th.function(
            inputs=[x_bat, y_bat, s_in_bat], 
            outputs=[cost_bat, s_out_bat])

        # Gradient calculations
        # We'll use this at some point for gradient checking
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
        pvalues = { n:m.get_value() for n, m in self.params.items() }

        # Now save params and matrices to file
        try:
            np.savez_compressed(outfile, p=p, **pvalues)
        except OSError as e:
            raise e
        else:
            if isinstance(outfile, str):
                stdout.write("Saved model parameters to {0}\n".format(outfile))

    def freshstate(self, batchsize):
        if batchsize > 0:
            return np.zeros([self.hyper.layers, batchsize, self.hyper.state_size], dtype=th.config.floatX)
        else:
            return np.zeros([self.hyper.layers, self.hyper.state_size], dtype=th.config.floatX)

    def calc_loss(self, dataset, startpos=0, batchsize=16, num_examples=0, init_state=None):
        step_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate(batchsize)

        if batchsize < 1:
            raise NotImplementedError("Single-sequence training is no longer available.")

        data_len = dataset.batchepoch(batchsize)
        valid_len = num_examples if num_examples else data_len
        errors = np.zeros(valid_len)

        # Use explicit indexing instead of fancy slicing so we can 
        # roll over properly
        data_pos = startpos
        for valid_pos in range(valid_len):
            xbatch, ybatch = dataset.batch(data_pos, batchsize)
            errors[valid_pos], step_state = self.err_bat(xbatch, ybatch, step_state)
            data_pos += 1
            # Advance position and overflow
            if data_pos >= data_len:
                data_pos = 0
                # Roll state vector on batch axis, to keep continuity
                step_state = np.roll(step_state, 1, axis=1)

        # Return total loss divided by number of characters in sample
        return np.sum(errors).item() / float(valid_len * batchsize * dataset.seq_len)

    def train(self, dataset, batchsize=16, num_examples=0, callback_every=1000, callback=None, init_state=None):
        """Train model on given dataset for num_examples, with optional 
        batch size.

        Optional callback function called after callback_every, with 
        model and current state as arguments.

        Inputs and outputs assumed to be numpy arrays (or equivalent)
        of 2 dimensions.

        If num_examples is 0, will train for full epoch.
        """

        # Batched training only
        if batchsize < 1:
            raise NotImplementedError("Single-sequence training is no longer available.")

        # First build training functions if not already done
        if not self._built_t:
            self._build_t()

        input_len = dataset.batchepoch(batchsize)
        train_len = num_examples if num_examples else input_len

        # Start with fresh state if none provided
        step_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate(batchsize)

        # Debug
        # print("Training with batchsize {0:d}, state shape {1}".format(batchsize, repr(step_state.shape)))

        # Use explicit indexing instead of fancy slicing so we can 
        # keep track, both for model status and checkpoint purposes
        for train_pos in range(train_len):
            # Learning step
            xbatch, ybatch = dataset.batch(self.pos, batchsize)
            step_state = self.train_step_bat(xbatch, ybatch, step_state, 
                self.hyper.learnrate, self.hyper.decay)

            # Advance position and overflow
            self.pos += 1
            if self.pos >= input_len:
                self.epoch += 1
                self.pos = 0
                # Roll state vector on batch axis, to keep continuity
                step_state = np.roll(step_state, 1, axis=1)

            # Optional callback
            if callback and callback_every and (train_pos + 1) % callback_every == 0:
                # Make sure to only pass a slice of state if batched
                callback(self, step_state[:,0,:])

        # Return final state
        return step_state

    def traintime(self, dataset, batchsize=16, pos=0, init_state=None):
        """Prints time for batch training step (default size 16).
        Input must be 3D tensor of matrices of one-hot vectors.
        """
        # Fresh state
        start_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate(batchsize)

        # Get slice
        xbatch, ybatch = dataset.batch(pos, batchsize)

        # Time training step
        time1 = time.time()
        self.train_step_bat(xbatch, ybatch, start_state, self.hyper.learnrate, self.hyper.decay)
        time2 = time.time()

        stdout.write(
            "Time for SGD/RMS learning batch of {0:d} sequences, {1:d} chars each: {2:.4f} ms\n".format(
            xbatch.shape[1], xbatch.shape[0], (time2 - time1) * 1000.0))

        # Time loss calc step
        time1 = time.time()
        self.err_bat(xbatch, ybatch, start_state)
        time2 = time.time()

        stdout.write("Time for loss calculation step of {0:d} chars: {1:.4f} ms\n".format(
            xbatch.shape[0], (time2 - time1) * 1000.0))

    def genchars(self, charset, numchars, init_state=None, seedch=None, use_max=False, temperature=1.0):
        """Generate string of characters from current model parameters.

        If use_max is True, will select most-likely character at each step.

        Probabilities can be optionally scaled by temperature during generation
        if use_max=False. 
        """

        # Fresh state
        start_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate(0)

        # Seed given or random character to start (as one-hot)
        if seedch:
            seedidx = charset.idxofchar(seedch)
        else:
            try:
                seedidx = charset.semirandomidx()
            except AttributeError:
                seedidx = charset.randomidx()

        seedvec = charset.onehot(seedidx)

        # Get generated sequence
        if use_max:
            idxs, end_state = self.gen_chars_max(numchars - 1, seedvec, start_state)
        else:
            idxs, end_state = self.gen_chars(numchars - 1, seedvec, start_state, temperature)

        # Convert to characters
        chars = [ charset.charatidx(np.argmax(i)) for i in idxs ]

        # Now construct string
        return charset.charatidx(np.argmax(seedvec)) + "".join(chars), end_state


