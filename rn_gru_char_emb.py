#!/usr/bin/env python3

import os, datetime, pickle, random, time
from sys import stdin, stdout, stderr
import numpy as np
import theano
import theano.tensor as T

class HyperParams:
    """Hyperparameters for GRU setup."""

    def __init__(self, vocab_size, state_size=128, layers=1, bptt_truncate=-1, learnrate=0.001, decay=0.9):
        self.vocab_size = vocab_size
        self.state_size = state_size
        self.layers = layers
        self.bptt_truncate = bptt_truncate
        self.learnrate = learnrate
        self.decay = decay


# TODO: add dropout between layers, *proper* embedding layer hooks
# TODO: make scaffolding for context windows (might need to involve charset)
class ModelParams:
    """Model parameter matrices for GRU setup.
    U and W are gate matrices, 3 per layer (reset gate, update gate, state gate).
    V translates back to vocab-sized vector for output (classification layer).
    """

    def __init__(self, hyper, epoch=0, pos=0, V=None, U=None, W=None, a=None, b=None, c=None):

        self.hyper = hyper
        self.epoch = epoch
        self.pos = pos

        # Randomly initialize matrices if not provided
        # U and W get 3 2D matrices per layer (reset and update gates plus hidden state)
        # NOTE: as truth values of numpy arrays are ambiguous, explicit isinstance() used instead
        # NOTE2: copy provided arrays due to weird interactions between numpy load and Theano
        tV = np.copy(V) if isinstance(V, np.ndarray) else np.random.uniform(
            -np.sqrt(1.0/hyper.vocab_size), np.sqrt(1.0/hyper.vocab_size), 
            (hyper.state_size, hyper.vocab_size))

        tU = np.copy(U) if isinstance(U, np.ndarray) else np.random.uniform(
            -np.sqrt(1.0/hyper.state_size), np.sqrt(1.0/hyper.state_size), 
            (hyper.layers*3, hyper.state_size, hyper.state_size))

        tW = np.copy(W) if isinstance(W, np.ndarray) else np.random.uniform(
            -np.sqrt(1.0/hyper.state_size), np.sqrt(1.0/hyper.state_size), 
            (hyper.layers*3, hyper.state_size, hyper.state_size))

        # Initialize bias matrices to zeroes
        # b gets 3x2D per layer, c is single 2D
        ta = np.copy(a) if isinstance(a, np.ndarray) else np.zeros(hyper.state_size)
        tb = np.copy(b) if isinstance(b, np.ndarray) else np.zeros((hyper.layers*3, hyper.state_size))
        tc = np.copy(c) if isinstance(c, np.ndarray) else np.zeros(hyper.vocab_size)

        # Shared variables
        self.V = theano.shared(name='V', value=tV.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=tU.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=tW.astype(theano.config.floatX))
        self.a = theano.shared(name='a', value=ta.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=tb.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=tc.astype(theano.config.floatX))

        # rmsprop parameters
        self.mV = theano.shared(name='mV', value=np.zeros_like(tV).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros_like(tU).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros_like(tW).astype(theano.config.floatX))
        self.ma = theano.shared(name='ma', value=np.zeros_like(ta).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros_like(tb).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros_like(tc).astype(theano.config.floatX))

        # Build Theano graph and add related attributes
        self.theano = {}
        stdout.write("Compiling Theano graph and functions...")
        stdout.flush()
        time1 = time.time()
        self.__build_t__()
        time2 = time.time()
        stdout.write("done!\nCompilation took {0:.3f} ms.\n".format((time2 - time1) * 1000.0))
        stdout.flush()

    def __build_t__(self):
        """Build Theano graph and define functions."""

        # Inputs
        x = T.ivector('x')
        y = T.ivector('y')
        s_in = T.matrix('s_in')

        # Constants(ish)
        layers = self.hyper.layers
        vocab_size = self.hyper.vocab_size
        state_size = self.hyper.state_size

        # Local bindings for convenience
        V, U, W, a, b, c = self.V, self.U, self.W, self.a, self.b, self.c
        xI = T.eye(vocab_size, vocab_size)

        # Forward propagation
        def forward_step(x_t, s_t):
            """Input vector x(t) and state matrix s(t)."""

            # Initialize state to return
            s_next = T.zeros_like(s_t)

            # Not taking shortcut anymore
            # Create one-hot vector from x_t using column of xI
            x_vec = xI[:,x_t]

            # Vocab-to-state
            # Should we do a nonlinearity here?
            # inout = T.tanh(V.dot(x_vec) + a)
            inout = V.dot(x_vec) + a

            # Loop over layers
            for layer in range(layers):
                # 3 matrices per layer
                L = layer * 3
                # Get previous state for this layer
                s_prev = s_t[layer]
                # Update gate
                z = T.nnet.hard_sigmoid(U[L].dot(inout) + W[L].dot(s_prev) + b[L])
                # Reset gate
                r = T.nnet.hard_sigmoid(U[L+1].dot(inout) + W[L+1].dot(s_prev) + b[L+1])
                # Candidate state
                h = T.tanh(U[L+2].dot(inout) + W[L+2].dot(r * s_prev) + b[L+2])
                # New state
                s_new = (T.ones_like(z) - z) * h + z * s_prev
                s_next = T.set_subtensor(s_next[layer], s_new)
                # Update for next layer or final output (might add dropout here later)
                inout = s_new

            # Final output
            # Theano's softmax returns matrix, and we just want the column
            o_t = T.nnet.softmax(V.T.dot(inout) + c)[0]
            o_norm = o_t / T.sum(o_t)
            return o_norm, s_next

        # Now get Theano to do the heavy lifting
        [o, s_seq], updates = theano.scan(
            forward_step, sequences=x, truncate_gradient=self.hyper.bptt_truncate,
            outputs_info=[None, dict(initial=s_in)])
        s_out = s_seq[-1]

        o_err = T.sum(T.nnet.categorical_crossentropy(o, y))
        # Should regularize at some point
        cost = o_err

        # Gradients
        dV = T.grad(cost, V)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        da = T.grad(cost, a)
        db = T.grad(cost, b)
        dc = T.grad(cost, c)

        # rmsprop parameter updates
        learnrate = T.scalar('learnrate')
        decayrate = T.scalar('decayrate')
        mV = decayrate * self.mV + (1 - decayrate) * dV ** 2
        mU = decayrate * self.mU + (1 - decayrate) * dU ** 2
        mW = decayrate * self.mW + (1 - decayrate) * dW ** 2
        ma = decayrate * self.ma + (1 - decayrate) * da ** 2
        mb = decayrate * self.mb + (1 - decayrate) * db ** 2
        mc = decayrate * self.mc + (1 - decayrate) * dc ** 2

        # Assign Theano-constructed functions to instance

        # Error
        self.err = theano.function([x, y, s_in], [cost, s_out])
        # Backpropagation
        # We'll use this at some point for gradient checking
        self.bptt = theano.function([x, y, s_in], [dV, dU, dW, da, db, dc])

        # Training step function
        self.train_step = theano.function(
            [x, y, s_in, theano.Param(learnrate, default=0.001), theano.Param(decayrate, default=0.9)],
            s_out,
            updates=[
                (V, V - learnrate * dV / T.sqrt(mV + 1e-6)),
                (U, U - learnrate * dU / T.sqrt(mU + 1e-6)),
                (W, W - learnrate * dW / T.sqrt(mW + 1e-6)),
                (a, a - learnrate * da / T.sqrt(ma + 1e-6)),
                (b, b - learnrate * db / T.sqrt(mb + 1e-6)),
                (c, c - learnrate * dc / T.sqrt(mc + 1e-6)),
                (self.mV, mV),
                (self.mU, mU),
                (self.mW, mW),
                (self.ma, ma),
                (self.mb, mb),
                (self.mc, mc)],
            name = 'train_step')

        # Predicted char probabilities (old version, reqires recursive sequence input)
        self.predict_prob = theano.function([x, s_in], [o, s_out])

        # Generate output sequence based on input char index and state (new version)
        x_in = T.iscalar('x_in')
        k = T.iscalar('k')
        rng = T.shared_randomstreams.RandomStreams(seed=int(
            self.V.get_value()[0,0] +
            self.U.get_value()[0,0,0] + 
            self.W.get_value()[0,0,0])) 

        def generate_step(x_t, s_t):
            # Do next step
            o_t1, s_t1 = forward_step(x_t, s_t)

            # Randomly choose by multinomial distribution
            o_rand = rng.multinomial(size=o_t1.shape, n=1, pvals=o_t1)

            # Now find selected index
            o_idx = T.argmax(o_rand).astype('int32')

            return o_idx, s_t1

        [o_chs, s_chs], genupdate = theano.scan(
            fn=generate_step,
            outputs_info=[dict(initial=x_in), dict(initial=s_in)],
            n_steps=k)
        s_ch = s_chs[-1]
        self.gen_chars = theano.function([k, x_in, s_in], [o_chs, s_ch], name='gen_chars', updates=genupdate)

        # Whew, I think we're done!
        # (Loss functions further down)

    @classmethod
    def loadfromfile(cls, infile):
        with np.load(infile) as f:
            # Load matrices
            p, V, U, W, a, b, c = f['p'], f['V'], f['U'], f['W'], f['a'], f['b'], f['c']

            # Extract hyperparams and position
            params = pickle.loads(p.tobytes())
            hyper, epoch, pos = params['hyper'], params['epoch'], params['pos']

            # Create instance
            model = cls(hyper, epoch, pos, V=V, U=U, W=W, a=a, b=b, c=c)
            if isinstance(infile, str):
                stderr.write("Loaded model parameters from {0}\n".format(infile))

            return model

    def savetofile(self, outfile):
        # Pickle non-matrix params into bytestring, then convert to numpy byte array
        pklbytes = pickle.dumps({'hyper': self.hyper, 'epoch': self.epoch, 'pos': self.pos}, 
            protocol=pickle.HIGHEST_PROTOCOL)
        p = np.fromstring(pklbytes, dtype=np.uint8)

        # Now save params and matrices to file
        try:
            np.savez(outfile, p=p, 
                V=self.V.get_value(), 
                U=self.U.get_value(), 
                W=self.W.get_value(), 
                a=self.a.get_value(), 
                b=self.b.get_value(), 
                c=self.c.get_value())
        except OSError as e:
            raise e
        else:
            if isinstance(outfile, str):
                stderr.write("Saved model parameters to {0}\n".format(outfile))

    def calc_total_loss(self, X, Y):
        step_state = self.freshstate()
        errors = np.zeros(len(X))

        for pos in range(len(X)):
            errors[pos], step_state = self.err(X[pos], Y[pos], step_state)

        return np.sum(errors).item()

    def calc_loss(self, X, Y):
        return self.calc_total_loss(X, Y) / float(X.size)

    def freshstate(self):
        return np.zeros([self.hyper.layers, self.hyper.state_size], dtype=theano.config.floatX)

    def train(self, inputs, outputs, num_epochs=1, num_pos=0, callback_every=1000, callback=None):
        """Train model on given inputs/outputs for given num_epochs for
        num_pos examples per cycle.

        Optional callback function called after callback_every, with 
        model as argument.

        Inputs and outputs assumed to be numpy arrays (or equivalent)
        of 2 dimensions.

        If num_epochs is 0, will only train on num_pos examples, starting
        from last pos.

        If num_epochs is > 0, will train on num_pos examples per epoch, 
        or full input length if num_pos is 0.
        """
        # Use explicit indexing so we can keep track, both for
        # checkpoint purposes, and to check against callback_every
        input_len = inputs.shape[0]

        if num_epochs:
            train_len = num_pos if num_pos > 0 else input_len

            for epoch in range(num_epochs):
                # Fresh state
                step_state = self.freshstate()

                # Loop over training data
                for pos in range(train_len):
                    # Learning step
                    step_state = self.train_step(inputs[pos], outputs[pos], step_state, 
                        self.hyper.learnrate, self.hyper.decay)

                    # Optional callback
                    if callback and callback_every and (epoch * input_len + pos) % callback_every == 0:
                        callback(self)

                    self.pos = pos + 1

                self.epoch += 1
                self.pos = 0
        else:
            step_state = self.freshstate()
            start_pos = self.pos
            end_pos = start_pos + num_pos if (start_pos +  num_pos) <= input_len else input_len

            # Loop over training data
            for pos in range(start_pos, end_pos):
                # Learning step
                step_state = self.train_step(inputs[pos], outputs[pos], step_state, 
                    self.hyper.learnrate, self.hyper.decay)
                
                # Optional callback
                if callback and callback_every and (pos - start_pos) % callback_every == 0:
                    callback(self)

                self.pos = pos + 1

    def genchars(self, charset, numchars, init_state=None):
        """Generate string of characters from current model parameters."""

        # Fresh state
        start_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate()

        # Seed random character to start
        seedch = charset.randomidx()

        # Get generated sequence
        idxs, end_state = self.gen_chars(numchars, seedch, start_state)

        # Now translate from indicies to characters, and construct string
        chars = [ charset.charatidx(i) for i in idxs ]
        return charset.charatidx(seedch) + "".join(chars), end_state

    def genchars_prob(self, charset, numchars, init_state=None, choose_max=False):
        """Generate string of characters from current model parameters, 
        using predicted next char probabilities (recursive version).
        """

        # Fresh state
        prev_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate()

        # Seed random character to start
        idxs = [charset.randomidx()]

        # Choose the probabilistic version or the most-likely one
        # Note: we don't feed in the updated state, because predict_prob looks at
        # the whole sequence we feed (back) in
        for _ in range(numchars):
            # Get probability vector of next char
            next_probs, next_state = self.predict_prob(idxs, prev_state)

            if choose_max:
                # Get most-likely next char
                next_idx = np.argmax(next_probs[-1])
            else:
                # Choose char from weighted random choice
                next_idx = np.random.choice(charset.vocab_size, p=next_probs[-1])

            # Append to list, and round we go
            idxs.append(next_idx)

        # Now translate from indicies to characters, and construct string
        chars = [ charset.charatidx(i) for i in idxs ]

        # We *do*, however, return the final updated state
        return "".join(chars), next_state

    # TODO: Non-Theano step function
    # TODO: Non-Theano char generator

    def traintime(self, inputvec, outputvec, init_state=None):
        """Prints time for single training step.
        Input should be single-dim vector.
        """
        # Fresh state
        start_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate()

        # Time training step
        time1 = time.time()
        self.train_step(inputvec, outputvec, start_state, self.hyper.learnrate, self.hyper.decay)
        time2 = time.time()

        stdout.write("Time for SGD/RMS learning step of {0:d} chars: {1:.4f} ms\n".format(
            len(inputvec), (time2 - time1) * 1000.0))

        # Time loss calc
        time1 = time.time()
        self.err(inputvec, outputvec, start_state)
        time2 = time.time()

        stdout.write("Time for loss calculation step of {0:d} chars: {1:.4f} ms\n".format(
            len(inputvec), (time2 - time1) * 1000.0))


class CharSet:
    """Character set with bidirectional mappings."""

    unknown_char='�'    # Standard Python replacement char for invalid Unicode values

    def __init__(self, chars, srcinfo=None):
        '''Creates a new CharSet object from a given sequence.
        Parameter chars should be a list, tuple, or set of characters.
        Parameter srcinfo is optional, to identify charset used for processing.
        '''
        self.srcinfo = srcinfo

        # Create temp list (mutable), starting with unknown
        # (We want unknown to be index 0 in case it shows up by accident later in arrays)
        charlist = [self.unknown_char]
        charlist.extend([ch for ch in chars if ch != self.unknown_char])

        # Create cross-mappings
        self._idx_to_char = { i:ch for i,ch in enumerate(charlist) }
        self._char_to_idx = { ch:i for i,ch in self._idx_to_char.items() }
        # The following *should* be zero
        self.unknown_idx = self._char_to_idx[self.unknown_char]

        # Now we can set vocab size
        self.vocab_size = len(self._char_to_idx)

        stderr.write("Initialized character set, size: {0:d}\n".format(self.vocab_size))

    def idxofchar(self, char):
        '''Returns index of char, or index of unknown replacement if index out of range.'''
        if char in self._char_to_idx:
            return self._char_to_idx[char]
        else:
            return self.unknown_idx

    def charatidx(self, idx):
        '''Returns character at idx, or unknown replacement if character not in set.'''
        if idx in self._idx_to_char:
            return self._idx_to_char[idx]
        else:
            return self.unknown_char

    def randomidx(self, allow_newline=False):
        '''Returns random character, excluding unknown_char.'''
        forbidden = [self.unknown_idx]
        if not allow_newline:
            forbidden.append(self.idxofchar('\n'))

        char = self.unknown_idx
        while char in forbidden:
            char = random.randrange(self.vocab_size)
        return char

        
class DataSet:
    """Preprocessed dataset, split into sequences and stored as arrays of character indexes."""

    # TODO: save/load arrays separately as .npz, and other attributes in dict

    def __init__(self, datastr, charset, seq_len=50, srcinfo=None, savedarrays=None):
        self.datastr = datastr
        self.charinfo = charset.srcinfo
        self.seq_len = seq_len
        self.srcinfo = srcinfo

        if savedarrays:
            stderr.write("Loading arrays...\n")

            self.x_array = savedarrays['x_array']
            self.y_array = savedarrays['y_array']

            stderr.write("Loaded arrays, x: {0} y: {1}\n".format(repr(self.x_array.shape), repr(self.y_array.shape)))

        else:
            stderr.write("Processing data string of {0:d} bytes...\n".format(len(datastr)))

            # Encode into charset indicies, skipping empty sequences
            x_sequences, y_sequences = [], []
            for pos in range(0, len(datastr)-1, seq_len):
                if pos+seq_len < len(datastr):
                    # Add normally while slices are full sequence length
                    x_sequences.append([ charset.idxofchar(ch) for ch in datastr[pos:pos+seq_len] ])
                    y_sequences.append([ charset.idxofchar(ch) for ch in datastr[pos+1:pos+seq_len+1] ])
                else:
                    # Pad otherwise-truncated final sequence with text from beginning
                    x_sequences.append([ charset.idxofchar(ch) for ch in (datastr[pos:] + datastr[:pos+seq_len-len(datastr)]) ])
                    y_sequences.append([ charset.idxofchar(ch) for ch in (datastr[pos+1:] + datastr[:pos+1+seq_len-len(datastr)]) ])

            # Encode sequences into arrays for training
            self.x_array = np.asarray(x_sequences, dtype='int32')
            self.y_array = np.asarray(y_sequences, dtype='int32')

            stderr.write("Initialized arrays, x: {0} y: {1}\n".format(repr(self.x_array.shape), repr(self.y_array.shape)))

    @staticmethod
    def loadfromfile(filename):
        """Loads data set from filename."""

        try:
            f = open(filename, 'rb')
        except OSError as e:
            stderr.write("Couldn't open data set file, error: {0}\n".format(e))
            return None
        else:
            try:
                dataset = pickle.load(f)
            except Exception as e:
                stderr.write("Couldn't load data set, error: {0}\n".format(e))
                return None
            else:
                stderr.write("Loaded data set from {0}\n".format(filename))
                return dataset
        finally:
            f.close()

    def savetofile(self, savedir):
        """Saves data set to file in savedir.
        Filename taken from srcinfo if possible, otherwise defaults to 'dataset.p'.
        Returns filename if successful, None otherwise.
        """
        # Create directory if necessary (won't throw exception if dir already exists)
        os.makedirs(savedir, exist_ok=True)

        if isinstance(self.srcinfo, str):
            filename = os.path.join(savedir, self.srcinfo + ".p")
        elif isinstance(self.srcinfo, dict) and 'name' in self.srcinfo:
            filename = os.path.join(savedir, self.srcinfo['name'] + ".p")
        else:
            filename = os.path.join(savedir, "dataset.p")

        try:
            f = open(filename, 'wb')
        except OSError as e:
            stderr.write("Couldn't open target file, error: {0}\n".format(e))
            return None
        else:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            stderr.write("Saved data set to {0}\n".format(filename))
            return filename
        finally:
            f.close()


class Checkpoint:
    """Checkpoint for model training."""

    def __init__(self, datafile, modelfile, cp_date, epoch, pos, loss):
        self.datafile = datafile
        self.modelfile = modelfile
        self.cp_date = cp_date
        self.epoch = epoch
        self.pos = pos
        self.loss = loss

    @classmethod
    def createcheckpoint(cls, savedir, datafile, modelparams, loss):
        """Creates and saves modelparams and pickled training checkpoint into savedir.
        Returns new checkpoint and filename if successful, or (None, None) otherwise.
        """

        # Create directory if necessary (won't throw exception if dir already exists)
        os.makedirs(savedir, exist_ok=True)

        # Determine filenames
        modeldatetime = datetime.datetime.now(datetime.timezone.utc)
        basefilename = modeldatetime.strftime("%Y-%m-%d-%H:%M:%S-UTC")

        # Save model file
        modelfilename = os.path.join(savedir, basefilename + ".npz")
        try:
            modelfile = open(modelfilename, 'wb')
        except OSError as e:
            stderr.write("Couldn't save model parameters to {0}!\nError: {1}\n".format(modelfilename, e))
            return None, None
        else:
            modelparams.savetofile(modelfile)
            stderr.write("Saved model parameters to {0}\n".format(modelfilename))

            # Create checkpoint
            cp = cls(datafile, modelfilename, modeldatetime, modelparams.epoch, modelparams.pos, loss)
            cpfilename = os.path.join(savedir, basefilename + ".p")

            # Save checkpoint
            try:
                cpfile = open(cpfilename, 'wb')
            except OSError as e:
                stderr.write("Couldn't save checkpoint to {0}!\nError: {1}\n".format(cpfilename, e))
                return None, None
            else:
                pickle.dump(cp, cpfile, protocol=pickle.HIGHEST_PROTOCOL)
                stderr.write("Saved checkpoint to {0}\n".format(cpfilename))
                return cp, cpfilename
            finally:
                cpfile.close()
        finally:
            modelfile.close()

    @classmethod
    def loadcheckpoint(cls, cpfile):
        """Loads checkpoint from saved file and returns checkpoint object."""
        try:
            f = open(cpfile, 'rb')
        except OSError as e:
            stderr.write("Couldn't open checkpoint file {0}!\nError: {1}\n".format(cpfile, e))
            return None
        else:
            try:
                stderr.write("Restoring checkpoint from file {0}...\n".format(cpfile))
                cp = pickle.load(f)
            except Exception as e:
                stderr.write("Error restoring checkpoint from file {0}:\n{1}\n".format(cpfile, e))
                return None
            else:
                return cp
        finally:
            f.close()

    def printstats(self, outfile):
        """Prints checkpoint stats to file-like object."""

        printstr = """Checkpoint date: {0}
        Dataset file: {1}
        Model file: {2}
        Epoch: {3:d}
        Position: {4:d}
        Loss: {5:.4f}
        """.format(
            self.cp_date.strftime("%Y-%m-%d %H:%M:%S %Z"), 
            self.datafile, 
            self.modelfile, 
            self.epoch, 
            self.pos, 
            self.loss)
        outfile.write(printstr)
        

class ModelState:
    """Model state, including hyperparamters, charset, last-loaded 
    checkpoint, dataset, and model parameters.

    Note: Checkpoint is automatically loaded when restoring from file, 
    but dataset and model parameters must be explicitly (re)loaded.
    """

    def __init__(self, chars, curdir, srcinfo=None, cpfile=None, 
        cp=None, datafile=None, data=None, modelfile=None, model=None):
        self.chars = chars
        self.srcinfo = srcinfo
        self.curdir = curdir
        self.cpfile = cpfile
        self.cp = cp
        self.datafile = datafile
        self.data = data
        self.modelfile = modelfile
        self.model = model

    def __getstate__(self):
        state = self.__dict__.copy()
        # References to checkpoint, dataset, and model params 
        # shouldn't be serialized here, so remove them
        state['cp'] = None
        state['data'] = None
        state['model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reload checkpoint, if present
        if self.cpfile:
            self.cp = Checkpoint.loadcheckpoint(self.cpfile)
            if self.cp:
                stderr.write("Loaded checkpoint from {0}\n".format(self.cpfile))
            else:
                stderr.write("Couldn't load checkpoint from {0}\n".format(self.cpfile))
                # Checkpoint is invalid, so don't use its file
                self.cpfile = None

    @classmethod
    def initfromsrcfile(cls, srcfile, usedir, *, seq_len=100, init_checkpoint=True, **kwargs):
        """Initializes a complete model based on given source textfile and hyperparameters.
        Creates initial checkpoint after model creation if init_checkpoint is True.
        Additional keyword arguments are passed to HyperParams.
        """
        
        # First, create directory if req'd
        try:
            os.makedirs(usedir, exist_ok=True)
        except OSError as e:
            stderr.write("Error creating directory {0}: {1}".format(srcfile, e))
            raise e
        
        # Next, read source file
        try:
            f = open(srcfile, 'r', encoding='utf-8')
        except OSError as e:
            stderr.write("Error opening source file {0}: {1}".format(srcfile, e))
            raise e
        else:
            datastr = f.read()
        finally:
            f.close()

        # Determine full path of working dir and base name of source file
        # Will be using these later on
        dirname = os.path.abspath(usedir)
        basename = os.path.basename(srcfile)

        # Now find character set
        charset = CharSet(set(datastr), srcinfo=(basename + "-chars"))

        # And set hyperparameters (additional keyword args passed through)
        hyperparams = HyperParams(charset.vocab_size, **kwargs)

        # Create dataset, and save
        dataset = DataSet(datastr, charset, seq_len=seq_len, srcinfo=(basename + "-data"))
        datafilename = dataset.savetofile(dirname)

        # Now we can initialize the state
        modelstate = cls(charset, dirname, srcinfo=(basename + "-state"), 
            datafile=datafilename, data=dataset)

        # And build the model, with optional checkpoint
        if init_checkpoint:
            modelstate.buildmodelparams(hyperparams, dirname)
        else:
            modelstate.buildmodelparams(hyperparams)

        # Save initial model state
        #modelstate.savetofile(dirname)
        # Already saved during buildmodelparams()

        return modelstate


    @staticmethod
    def loadfromfile(filename):
        """Loads model state from filename.
        Note: dataset and model params can be restored from last checkpoint
        after loading model state using restorefrom().
        """

        try:
            f = open(filename, 'rb')
        except OSError as e:
            stderr.write("Couldn't open model state file, error: {0}\n".format(e))
        else:
            try:
                modelstate = pickle.load(f)
            except Exception as e:
                stderr.write("Couldn't load model state, error: {0}\n".format(e))
                return None
            else:
                stderr.write("Loaded model state from {0}\n".format(filename))
                return modelstate
        finally:
            f.close()

    def savetofile(self, savedir):
        """Saves model state to file in savedir.
        Filename taken from srcinfo if possible, otherwise defaults to 'modelstate.p'.
        Returns filename if successful, None otherwise.
        """

        if savedir:
            usedir = savedir
        else:
            if self.curdir:
                usedir = self.curdir
            else:
                raise FileNotFoundError('No directory specified!')

        # Create directory if necessary (won't throw exception if dir already exists)
        try:
            os.makedirs(usedir, exist_ok=True)
        except OSError as e:
            raise e
        else:
            if isinstance(self.srcinfo, str):
                filename = os.path.join(usedir, self.srcinfo + ".p")
            elif isinstance(self.srcinfo, dict) and 'name' in self.srcinfo:
                filename = os.path.join(usedir, self.srcinfo['name'] + ".p")
            else:
                filename = os.path.join(usedir, "modelstate.p")

            try:
                f = open(filename, 'wb')
            except OSError as e:
                stderr.write("Couldn't open target file, error: {0}\n".format(e))
                return None
            else:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
                stderr.write("Saved model state to {0}\n".format(filename))
                return filename
            finally:
                f.close()

    def loaddata(self, filename=None):
        """Attempts to load dataset first from given file, 
        then from current data file, then from current checkpoint (or file).
        """
        if filename:
            openfile = filename
        elif self.datafile:
            openfile = self.datafile
        elif self.cp:
            openfile = self.cp.datafile
        elif self.cpfile:
            # Try loading from file
            self.cp = Checkpoint.loadcheckpoint(self.cpfile)
            if self.cp:
                openfile = self.cp.datafile
            else:
                # Still didn't work, clear file listing (it's obviously bad)
                self.cpfile = None
                return False
        else:
            # No checkpoint and no file means no-go
            stderr.write("No checkpoint file to load!\n")
            return False

        # Load data now that filename is established
        self.data = DataSet.loadfromfile(openfile)
        if self.data:
            self.datafile = openfile
            return True
        else:
            return False

    def loadmodel(self, filename=None):
        """Attempts to load model parameters first from given file, 
        then from current model file, then from current checkpoint (or file).
        """
        if filename:
            openfile = filename
        elif self.modelfile:
            openfile = self.modelfile
        elif self.cp:
            openfile = self.cp.modelfile
        elif self.cpfile:
            # Try loading from file
            self.cp = Checkpoint.loadcheckpoint(self.cpfile)
            if self.cp:
                openfile = self.cp.modelfile
            else:
                # Still didn't work, clear file listing (it's obviously bad)
                self.cpfile = None
                return False
        else:
            # No checkpoint and no file means no-go
            stderr.write("No checkpoint file to load!\n")
            return False

        # Load model now that filename is established
        self.model = ModelParams.loadfromfile(openfile)
        if self.model:
            self.modelfile = openfile
            return True
        else:
            return False

    def restorefrom(self, checkpoint=None):
        """Restores dataset and model params from specified checkpoint.
        Defaults to stored checkpoint if none provided.
        """

        if checkpoint:
            # Checkpoint given, use that
            cp = checkpoint
        elif self.cp:
            # Try stored checkpoint
            cp = self.cp
        elif self.cpfile:
            # Try loading checkpoint from file
            self.cp = Checkpoint.loadcheckpoint(self.cpfile)
            if self.cp:
                cp = self.cp
            else:
                # Still didn't work, clear file listing (it's obviously bad)
                self.cpfile = None
                return False
        else:
            # No checkpoint and no file means no-go
            stderr.write("No checkpoint file to load!\n")
            return False

        # Load data and model, return True only if both work
        # Passing checkpoint's data/model filenames, overriding 
        # those already stored in model state
        if self.loaddata(cp.datafile) and self.loadmodel(cp.modelfile):
            return True
        else:
            return False

    def newcheckpoint(self, loss, savedir=None):
        """Creates new checkpoint with current datafile and model params.
        Defaults to saving into current working directory.
        """

        # Make sure we have prereqs
        if not self.datafile:
            stderr.write("Can't create checkpoint: no data file specified.\n")
            return False
        if not self.model:
            stderr.write("Can't create checkpoint: no model loaded.\n")
            return False

        # Use specified dir if provided, otherwise curdir
        usedir = savedir if savedir else self.curdir

        # Try creating checkpoint
        cp, cpfile = Checkpoint.createcheckpoint(usedir, self.datafile, self.model, loss)
        if cp:
            self.cp = cp
            self.cpfile = cpfile
            # Also save ourselves
            savefile = self.savetofile(usedir)
            if savefile:
                #stderr.write("Saved model state to {0}\n".format(savefile))
                return True
            else:
                return False
        else:
            return False

    def builddataset(self, datastr, seq_len=100, srcinfo=None):
        """Builds new dataset from string and saves to file in working directory."""

        # Build dataset from string
        self.data = DataSet(datastr, self.chars, seq_len, srcinfo)

        # Save to file in working directory
        self.datafile = self.data.savetofile(self.curdir)

        # Return true if both operations succeed
        if self.data and self.datafile:
            return True
        else:
            return False

    def buildmodelparams(self, hyper, checkpointdir=None):
        """Builds model parameters from given hyperparameters and charset size.
        Optionally saves checkpoint immediately after building if path specified.
        """

        self.model = ModelParams(hyper)

        if checkpointdir:
            # Get initial loss estimate
            stderr.write("Calculating initial loss estimate...\n")
            loss_len = 1000 if len(self.data.x_array) >= 1000 else len(self.data.x_array)
            loss = self.model.calc_loss(self.data.x_array[:loss_len], self.data.y_array[:loss_len])
            stderr.write("Initial loss: {0:.3f}\n".format(loss))

            # Take checkpoint
            self.newcheckpoint(loss, savedir=checkpointdir)

    def trainmodel(self, num_rounds=1, round_epochs=1, print_every=1000):
        """Train loaded model for num_rounds of round_epochs passes, printing
        progress every print_every examples, calculating loss and creating 
        a checkpoint after each round.
        """

        # Make sure we have model and data loaded
        if not self.data or not self.model:
            stderr.write("Dataset and model parameters must be loaded before training!\n")
            return False

        # Progress callback
        progress = printprogress(self.chars)

        # Train for num_rounds
        for roundnum in range(num_rounds):
            # Train for round_epochs...
            self.model.train(
                self.data.x_array, 
                self.data.y_array,
                num_epochs=round_epochs,
                callback=progress,
                callback_every=print_every)

            # Calc loss
            loss = self.model.calc_loss(self.data.x_array, self.data.y_array)
            stderr.write("Previous loss: {0:.3f}, current loss: {1:.3f}\n".format(self.cp.loss, loss))

            # Adjust learning rate if necessary
            if loss > self.cp.loss:
                self.model.hyper.learnrate *= 0.5
                stderr.write("Loss increased, adjusted learning rate to {0:.3f}".format(self.model.hyper.learnrate))

            # Take checkpoint and print stats
            self.newcheckpoint(loss)
            self.cp.printstats(stdout)

        stdout.write("Completed {0:d} rounds of {1:d} epochs each.\n".format(num_rounds, round_epochs))


# Unattached functions

def printprogress(charset):
    def retfunc (model):
        print("--------\n")
        print("Time: {0}".format(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")))
        print("Epoch: {0}, pos: {1}".format(model.epoch, model.pos))
        print("Generated 100 chars:\n")
        genstr, _ = model.genchars(charset, 100)
        print(genstr + "\n")
    return retfunc

# TODO: Non-Theano sigmoid
# TODO: Non-Theano softmax