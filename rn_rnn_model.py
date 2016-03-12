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
    # Parameter and rmsprop cache matrix names
    pnames = []
    mnames = []

    def __init__(self, hyper, epoch=0, pos=0):
        self.hyper = hyper
        self.epoch = epoch
        self.pos = pos
        self._built_g = False
        self._built_t = False

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

        # Scalar training parameters
        learnrate = T.scalar('learnrate')
        decayrate = T.scalar('decayrate')

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
        # Costs
        [o, s_seq], _ = th.scan(
            single_step, 
            sequences=x, 
            truncate_gradient=self.hyper.bptt_truncate,
            outputs_info=[None, dict(initial=s_in)])
        s_out = s_seq[-1]
        o_errs = T.nnet.categorical_crossentropy(o, y)
        # Should regularize at some point
        cost = T.sum(o_errs)

        # Gradients
        dparams = [ T.grad(cost, p) for p in self.params ]

        # rmsprop parameter updates
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

    # Old, non-batched version
    def _calc_loss_old(self, X, Y, init_state=None):
        step_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate()
        errors = np.zeros(len(X))

        # Use explicit indexing so a) we can feed the state back in, 
        # and b) more efficiently store returned errors
        for pos in range(len(X)):
            errors[pos], step_state = self.err(X[pos], Y[pos], step_state)

        # Return total loss divided by number of characters in sample
        return np.sum(errors).item() / float(X.size / X.shape[-1])

    def calc_loss(self, dataset, startpos, batchsize=0, num_examples=0, init_state=None):
        step_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate(batchsize)

        if batchsize == 0:
            # Get wraparound slices of dataset, since calc_loss doesn't update pos
            x_slice, y_slice = dataset.slices(startpos, num_examples)

            # Calculate loss old way with blank state
            return self.model._calc_loss_old(x_slice, y_slice, init_state=step_state)
        else:
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

    def train(self, dataset, batchsize=0, num_examples=0, callback_every=1000, callback=None, init_state=None):
        """Train model on given dataset for num_examples, with optional 
        batch size.

        Optional callback function called after callback_every, with 
        model and current state as arguments.

        Inputs and outputs assumed to be numpy arrays (or equivalent)
        of 2 dimensions.

        If num_examples is 0, will train for full epoch.
        """

        # First build training functions if not already done
        if not self._built_t:
            self._build_t()

        input_len = dataset.batchepoch(batchsize) if batchsize > 0 else dataset.data_len
        train_len = num_examples if num_examples else input_len

        # Start with fresh state if none provided
        step_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate(batchsize)

        # Debug
        # print("Training with batchsize {0:d}, state shape {1}".format(batchsize, repr(step_state.shape)))

        # Use explicit indexing instead of fancy slicing so we can 
        # keep track, both for model status and checkpoint purposes
        for train_pos in range(train_len):
            # Learning step
            if batchsize > 0:
                xbatch, ybatch = dataset.batch(self.pos, batchsize)
                step_state = self.train_step_bat(xbatch, ybatch, step_state, 
                    self.hyper.learnrate, self.hyper.decay)
            else:
                step_state = self.train_step(dataset.x_onehots[self.pos], dataset.y_onehots[self.pos], 
                    step_state, self.hyper.learnrate, self.hyper.decay)

            # Advance position and overflow
            self.pos += 1
            if self.pos >= input_len:
                self.epoch += 1
                self.pos = 0
                if batchsize > 0:
                    # Roll state vector on batch axis, to keep continuity
                    step_state = np.roll(step_state, 1, axis=1)
                else:
                    step_state = self.freshstate(batchsize)

            # Optional callback
            if callback and callback_every and (train_pos + 1) % callback_every == 0:
                # Make sure to only pass a slice of state if batched
                if batchsize > 0:
                    callback(self, step_state[:,0,:])
                else:
                    callback(self, step_state)

        # Return final state
        return step_state

    def traintime(self, inputmat, outputmat, init_state=None):
        """Prints time for single training step.
        Input must be matrix of one-hot vectors.
        """
        # Fresh state
        start_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate()

        # Time training step
        time1 = time.time()
        self.train_step(inputmat, outputmat, start_state, self.hyper.learnrate, self.hyper.decay)
        time2 = time.time()

        stdout.write("Time for SGD/RMS learning step of {0:d} chars: {1:.4f} ms\n".format(
            len(inputmat), (time2 - time1) * 1000.0))

        # Time loss calc
        time1 = time.time()
        self.err(inputmat, outputmat, start_state)
        time2 = time.time()

        stdout.write("Time for loss calculation step of {0:d} chars: {1:.4f} ms\n".format(
            len(inputmat), (time2 - time1) * 1000.0))

    def batchtime(self, intensor, outtensor, init_state=None):
        """Prints time for batch training step (default size 16).
        Input must be 3D tensor of matrices of one-hot vectors.
        """
        # Fresh state
        start_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate(intensor.shape[1])

        # Time training step
        time1 = time.time()
        self.train_step_bat(intensor, outtensor, start_state, self.hyper.learnrate, self.hyper.decay)
        time2 = time.time()

        stdout.write(
            "Time for SGD/RMS learning batch of {0:d} sequences, {1:d} chars each: {2:.4f} ms\n".format(
            intensor.shape[1], intensor.shape[0], (time2 - time1) * 1000.0))

        # Time loss calc
        # NOTE: uses only matrix from first part of batch (batched error not yet implemented)
        time1 = time.time()
        self.err_bat(intensor, outtensor, start_state)
        time2 = time.time()

        stdout.write("Time for loss calculation step of {0:d} chars: {1:.4f} ms\n".format(
            intensor.shape[0], (time2 - time1) * 1000.0))

    def genchars(self, charset, numchars, init_state=None, seedch=None, use_max=False, temperature=1.0):
        """Generate string of characters from current model parameters.

        If use_max is True, will select most-likely character at each step.

        Probabilities can be optionally scaled by temperature during generation
        if use_max=False. 
        """

        # Fresh state
        start_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate()

        # Seed given or random character to start (as one-hot)
        if seedch:
            seedidx = charset.idxofchar(seedch)
        else:
            seedidx = charset.semirandomidx()

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

    # Theano-generated model-dependent functions
    def train_step(self, *args, **kwargs):
        pass
    def train_step_bat(self, *args, **kwargs):
        pass
    def errs(self, *args, **kwargs):
        pass
    def errs_bat(self, *args, **kwargs):
        pass
    def err(self, *args, **kwargs):
        pass
    def err_bat(self, *args, **kwargs):
        pass
    def grad(self, *args, **kwargs):
        pass
    def grad_bat(self, *args, **kwargs):
        pass
    def gen_chars(self, *args, **kwargs):
        pass
    def gen_chars_max(self, *args, **kwargs):
        pass

