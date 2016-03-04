#!/usr/bin/env python3

import os, datetime, pickle
from sys import stdin, stdout, stderr
import numpy as np
import theano
import theano.tensor as T

class HyperParams:
    """Hyperparameters for GRU setup."""

    def __init__(self, state_size=128, layers=1, seq_len=100, bptt_truncate=-1):
        self.state_size = state_size
        self.layers = layers
        self.seq_len = seq_len
        self.bptt_truncate = bptt_truncate


class ModelParams:
    """Model parameter matrices for GRU setup."""

    def __init__(self, state_size, vocab_size, layers=1, U=None, V=None, W=None, b=None, c=None):
        self.state_size = state_size
        self.vocab_size = vocab_size
        self.layers = layers
        # Randomly initialize matrices if not provided
        # U and W get 3 2D matrices per layer (reset and update gates plus hidden state)
        self.U = U if U else np.random.uniform(-np.sqrt(1.0/vocab_size), np.sqrt(1.0/vocab_size), (layers*3, state_size, vocab_size))
        self.V = V if V else np.random.uniform(-np.sqrt(1.0/state_size), np.sqrt(1.0/state_size), (vocab_size, state_size))
        self.W = W if W else np.random.uniform(-np.sqrt(1.0/state_size), np.sqrt(1.0/state_size), (layers*3, state_size, state_size))
        # Initialize bias matrices to zeroes
        # b gets 3x2D per layer, c is single 2D
        self.b = b if b else np.zeros((layers*3, state_size))
        self.c = c if c else np.zeros(vocab_size)

    @classmethod
    def loadfromfile(cls, infile):
        with np.load(infile) as f:
            # Load matrices
            U, V, W, b, c = f['U'], f['V'], f['W'], f['b'], f['c']

            # Infer state, vocab, and layer size
            layers, state_size, vocab_size = U.shape[0] / 3, U.shape[1], U.shape[2]

            # Create instance
            model = cls(state_size, vocab_size, layers, U, V, W, b, c)
            if isinstance(infile, str):
                stderr.write("Loaded model parameters from {0:s}\n".format(infile))

            return model

    def savetofile(self, outfile):
        try:
            np.savez(outfile, U=self.U, V=self.V, W=self.W, b=self.b, c=self.c)
        except OSError as e:
            raise e
        else:
            if isinstance(outfile, str):
                stderr.write("Saved model parameters to {0:s}\n".format(outfile))


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

            stderr.write("Loaded arrays, x: {0:s} y: {1:s}\n".format(repr(self.x_array.shape), repr(self.y_array.shape)))

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

            stderr.write("Initialized arrays, x: {0:s} y: {1:s}\n".format(repr(self.x_array.shape), repr(self.y_array.shape)))

    @staticmethod
    def loadfromfile(filename):
        """Loads data set from filename."""

        try:
            f = open(filename, 'rb')
        except OSError as e:
            stderr.write("Couldn't open data set file, error: {0:s}\n".format(e))
            return None
        else:
            try:
                dataset = pickle.load(f)
            except Exception as e:
                stderr.write("Couldn't load data set, error: {0:s}\n".format(e))
                return None
            else:
                stderr.write("Loaded data set from {0:s}\n".format(filename))
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
            stderr.write("Couldn't open target file, error: {0:s}\n".format(e))
            return None
        else:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
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
    def createcheckpoint(cls, savedir, datafile, modelparams, epoch, pos, loss):
        """Creates and saves modelparams and pickled training checkpoint into savedir.
        Returns new checkpoint and filename if successful, or (None, None) otherwise.
        """

        # Create directory if necessary (won't throw exception if dir already exists)
        os.makedirs(savedir, exist_ok=True)

        # Determine filenames
        modeldatetime = datetime.datetime.now(datetime.timezone.utc)
        basefilename = modeldatetime.strftime("%Y-%m-%d-%H-%M-%S-%Z")

        # Save model file
        modelfilename = os.path.join(savedir, basefilename + ".npz")
        try:
            modelfile = open(modelfilename, 'wb')
        except OSError as e:
            stderr.write("Couldn't save model parameters to {0:s}!\nError: {1:s}\n".format(modelfilename, e))
            return None, None
        else:
            modelparams.savetofile(modelfile)
            stderr.write("Saved model parameters to {0:s}\n".format(modelfilename))

            # Create checkpoint
            cp = cls(datafile, modelfilename, modeldatetime, epoch, pos, loss)
            cpfilename = os.path.join(savedir, basefilename + ".p")

            # Save checkpoint
            try:
                cpfile = open(cpfilename, 'wb')
            except OSError as e:
                stderr.write("Couldn't save checkpoint to {0:s}!\nError: {1:s}\n".format(cpfilename, e))
                return None, None
            else:
                pickle.dump(cp, cpfile, protocol=pickle.HIGHEST_PROTOCOL)
                stderr.write("Saved checkpoint to {0:s}\n".format(cpfilename))
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
            stderr.write("Couldn't open checkpoint file {0:s}!\nError: {1:s}\n".format(cpfile, e))
            return None
        else:
            try:
                cp = pickle.load(f)
            except Exception as e:
                stderr.write("Error restoring checkpoint from file {0:s}:\n{1:s}\n".format(cpfile, e))
                return None
            else:
                stderr.write("Restored checkpoint from file {0:s}\n".format(cpfile))
                return cp
        finally:
            f.close()

    def printstats(self, outfile):
        """Prints checkpoint stats to file-like object."""

        printstr = """Checkpoint date: {0:s}
        Dataset file: {1:s}
        Model file: {2:s}
        Epoch: {3:s}
        Position: {4:s}
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

    def __init__(self, hyper, chars, curdir, srcinfo=None, cpfile=None, 
        cp=None, datafile=None, data=None, modelfile=None, model=None):
        self.hyper = hyper
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
            self.cp = Checkpoint.loadcheckpoint(cpfile)
            if self.cp:
                stderr.write("Loaded checkpoint from {0:s}\n".format(self.cpfile))
            else:
                stderr.write("Couldn't load checkpoint from {0:s}\n".format(self.cpfile))
                # Checkpoint is invalid, so don't use its file
                self.cpfile = None

    @classmethod
    def initfromsrcfile(cls, srcfile, usedir, *, init_checkpoint=True, **kwargs):
        """Initializes a complete model based on given source textfile and hyperparameters.
        Creates initial checkpoint after model creation if init_checkpoint is True.
        Additional keyword arguments are passed to HyperParams.
        """
        
        # First, create directory if req'd
        try:
            os.makedirs(usedir, exist_ok=True)
        except OSError as e:
            stderr.write("Error creating directory {}: {}".format(srcfile, e))
            raise e
        
        # Next, read source file
        try:
            f = open(srcfile, 'r', encoding='utf-8')
        except OSError as e:
            stderr.write("Error opening source file {}: {}".format(srcfile, e))
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
        hyperparams = HyperParams(**kwargs)

        # Create dataset, and save
        dataset = DataSet(datastr, charset, seq_len=hyperparams.seq_len, srcinfo=(basename + "-data"))
        datafilename = dataset.savetofile(dirname)

        # Now we can initialize the state
        modelstate = cls(charset, hyperparams, dirname, srcinfo=(basename + "-state"), 
            datafile=datafilename, data=dataset)

        # And build the model, with optional checkpoint
        modelstate.buildmodelparams(init_checkpoint)

        return modelstate


    @staticmethod
    def loadfromfile(filename):
        """Loads model state from filename."""

        try:
            f = open(filename, 'rb')
        except OSError as e:
            stderr.write("Couldn't open model state file, error: {0:s}\n".format(e))
        else:
            try:
                modelstate = pickle.load(f)
            except Exception as e:
                stderr.write("Couldn't load model state, error: {0:s}\n".format(e))
                return None
            else:
                stderr.write("Loaded model state from {0:s}\n".format(filename))
                return modelstate
        finally:
            f.close()

    def savetofile(self, savedir=self.curdir):
        """Saves model state to file in savedir.
        Filename taken from srcinfo if possible, otherwise defaults to 'modelstate.p'.
        Returns filename if successful, None otherwise.
        """

        if not savedir:
            raise FileNotFoundError('No directory specified!')

        # Create directory if necessary (won't throw exception if dir already exists)
        try:
            os.makedirs(savedir, exist_ok=True)
        except OSError as e:
            raise e
        else:
            if isinstance(self.srcinfo, str):
                filename = os.path.join(savedir, self.srcinfo + ".p")
            elif isinstance(self.srcinfo, dict) and 'name' in self.srcinfo:
                filename = os.path.join(savedir, self.srcinfo['name'] + ".p")
            else:
                filename = os.path.join(savedir, "modelstate.p")

            try:
                f = open(filename, 'wb')
            except OSError as e:
                stderr.write("Couldn't open target file, error: {0:s}\n".format(e))
                return None
            else:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
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

    def newcheckpoint(self, epoch, pos, loss):
        """Creates new checkpoint with current datafile and model params."""

        # Make sure we have prereqs
        if not self.datafile:
            stderr.write("Can't create checkpoint: no data file specified.\n")
            return False
        if not self.model:
            stderr.write("Can't create checkpoint: no model loaded.\n")
            return False

        # Try creating checkpoint
        cp, cpfile = Checkpoint.createcheckpoint(self.datafile, self.model, epoch, pos, loss)
        if cp:
            self.cp = cp
            self.cpfile = cpfile
            return True
        else:
            return False

    def builddataset(self, datastr, srcinfo=None):
        """Builds new dataset from string and saves to file in working directory."""

        # Build dataset from string
        self.data = DataSet(datastr, self.chars, self.hyper.seq_len, srcinfo)

        # Save to file in working directory
        self.datafile = self.data.savetofile(self.curdir)

        # Return true if both operations succeed
        if self.data and self.datafile:
            return True
        else:
            return False

    def buildmodelparams(self, checkpoint=False):
        """Builds model parameters from current hyperparameters and charset size.
        Optionally saves checkpoint immediately after building.
        """

        self.model = ModelParams(self.hyper.state_size, self.chars.vocab_size, self.hyper.layers)

        if checkpoint:
            self.newcheckpoint(0, 0, 0)

