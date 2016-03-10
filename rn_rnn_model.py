# Python module dependencies
import time
from sys import stdin, stdout, stderr
import numpy as np

class ModelParams:
    """Base class for RNN variants.
    NOTE: Not intended to be instantiated!
    """

    def __init__(self, hyper, epoch=0, pos=0):
        self.hyper = hyper
        self.epoch = epoch
        self.pos = pos

    @classmethod
    def loadfromfile(cls, *args, **kwargs):
        pass
        
    def savetofile(self, *args, **kwargs):
        pass

    def calc_loss(self, X, Y, init_state=None):
        step_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate()
        errors = np.zeros(len(X))

        # Use explicit indexing so a) we can feed the state back in, 
        # and b) more efficiently store returned errors
        for pos in range(len(X)):
            errors[pos], step_state = self.err(X[pos], Y[pos], step_state)

        # Return total loss divided by number of characters in sample
        return np.sum(errors).item() / float(X.size / X.shape[-1])

    def train(self, dataset, batchsize=0, num_examples=1000, callback_every=1000, callback=None, init_state=None):
        """Train model on given dataset for num_examples, with optional 
        batch size.

        Optional callback function called after callback_every, with 
        model and current state as arguments.

        Inputs and outputs assumed to be numpy arrays (or equivalent)
        of 2 dimensions.

        If num_examples is 0, will train for full epoch.
        """
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
        self.err(intensor[:,0,:], outtensor[:,0,:], start_state[:,0,:])
        time2 = time.time()

        stdout.write("Time for loss calculation step of {0:d} chars: {1:.4f} ms\n".format(
            intensor.shape[0], (time2 - time1) * 1000.0))

    def genchars(self, charset, numchars, init_state=None, use_max=False, temperature=0.0):
        """Generate string of characters from current model parameters.

        If use_max is True, will select most-likely character at each step.
        Probabilities can be optionally scaled by temperature during generation 
        if use_max=False.
        """

        # Fresh state
        start_state = init_state if isinstance(init_state, np.ndarray) else self.freshstate()

        # Seed random character to start (as one-hot)
        seedvec = charset.randomonehot()

        # Get generated sequence
        if use_max:
            if temperature > 0.0:
                idxs, end_state = self.gen_chars_max_temp(numchars - 1, seedvec, start_state, temperature)
            else:
                idxs, end_state = self.gen_chars_max(numchars - 1, seedvec, start_state)
        else:
            if temperature > 0.0:
                idxs, end_state = self.gen_chars_temp(numchars - 1, seedvec, start_state, temperature)
            else:
                idxs, end_state = self.gen_chars(numchars - 1, seedvec, start_state)

        # Convert to characters
        chars = [ charset.charatidx(np.argmax(i)) for i in idxs ]

        # Now construct string
        return charset.charatidx(np.argmax(seedvec)) + "".join(chars), end_state

    # TODO: Non-Theano step function
    # TODO: Non-Theano char generator

    # Python model-dependent functions
    def freshstate(self):
        pass

    # Theano-generated model-dependent functions
    def train_step(self, *args, **kwargs):
        pass
    def train_step_bat(self, *args, **kwargs):
        pass
    def errs(self, *args, **kwargs):
        pass
    def err(self, *args, **kwargs):
        pass
    def grad(self, *args, **kwargs):
        pass
    def gen_chars(self, *args, **kwargs):
        pass
    def gen_chars_temp(self, *args, **kwargs):
        pass
    def gen_chars_max(self, *args, **kwargs):
        pass

