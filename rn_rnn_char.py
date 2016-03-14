#!/usr/bin/env python3

#   Copyright 2016 Raymond Neilson
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# Module dependencies
import os, datetime, pickle, random, time
from sys import stdin, stdout, stderr
from math import log
import numpy as np
import theano as th
import theano.tensor as T

# Other files in this module
from rn_gru_resize import GRUResize
from rn_gru_encode import GRUEncode


class HyperParams:
    """Hyperparameters for GRU setup."""

    def __init__(self, vocab_size, state_size=128, layers=1, bptt_truncate=-1, 
        learnrate=0.001, decay=0.95, regcost=0.1):
        self.vocab_size = vocab_size
        self.state_size = state_size
        self.layers = layers
        self.bptt_truncate = bptt_truncate
        self.learnrate = learnrate
        self.decay = decay
        self.regcost = regcost

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # For backwards compatibility with models without regularization
        if 'regcost' not in state:
            self.regcost = 0.0


# TODO (maybe): change to let CharSet get frequencies from strings
class CharSet:
    """Character set with bidirectional mappings."""

    unknown_char='�'    # Standard Python replacement char for invalid Unicode values

    def __init__(self, datastr, srcinfo=None):
        '''Creates a new CharSet object from a given sequence.
        Parameter datastr should be a string or list of chars.
        Parameter srcinfo is optional, to identify charset used for processing.
        '''
        self.srcinfo = srcinfo

        # Find set of chars in supplied sequence
        # TODO: sort by frequency?
        chars = set(datastr)

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

        # Find characters that begin lines
        self.findlinestarts(datastr)

        stderr.write("Initialized character set, size: {0:d}\n".format(self.vocab_size))

    @classmethod
    def _linefinder(cls, datastr):
        '''Iterator over datastr, finding characters that start lines.'''
        start = 0
        maxidx = len(datastr) - 1
        while True:
            start = datastr.find('\n', start)
            if start == -1:
                return
            elif start < maxidx and datastr[start+1] not in '\n\r�':
                yield datastr[start+1]
            start += 1

    def findlinestarts(self, datastr):
        '''Finds characters that begin a line and stores as list.'''
        # linestartchars = set(self._linefinder(datastr))

        # Keep order of found chars
        foundchars = set()
        self._line_start_chars = []
        self._line_start_idxs = []
        for ch in self._linefinder(datastr):
            if ch not in foundchars:
                foundchars.add(ch)
                self._line_start_chars.append(ch)
                self._line_start_idxs.append(self.idxofchar(ch))

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

    def onehot(self, idx):
        '''Returns character at idx encoded as one-hot vector.'''
        vec = np.zeros(self.vocab_size, dtype=th.config.floatX)
        vec[idx] = 1.0
        return vec

    def encodeonehots(self, encodestr):
        '''Encodes string as matrix of one-hot vectors.'''

        # First get indices
        idxs = [ self.idxofchar(ch) for ch in encodestr ]

        # Now build onehots
        return np.eye(self.vocab_size, dtype=th.config.floatX)[idxs]

    def randomidx(self, allow_newline=False):
        '''Returns random character, excluding unknown_char.'''
        forbidden = {self.unknown_idx}
        if not allow_newline:
            forbidden.add(self.idxofchar('\n'))

        # Make sure we don't return an unknown char
        idx = self.unknown_idx
        while idx in forbidden:
            idx = random.randrange(self.vocab_size)

        return idx

    def semirandomidx(self):
        '''Returns random character from line-start list.'''
        return self._line_start_idxs[random.randrange(len(self._line_start_idxs))]

        
class DataSet:
    """Preprocessed dataset, split into sequences and stored as arrays of character indexes."""

    # TODO: save/load arrays separately as .npz, and other attributes in dict

    def __init__(self, datastr, charset, seq_len=50, srcinfo=None, savedarrays=None):
        self.datastr = datastr
        self.charinfo = charset.srcinfo
        self.charsize = charset.vocab_size
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
            if self.charsize <= 256:
                usetype = 'int8'
            else:
                usetype = 'int32'
            self.x_array = np.asarray(x_sequences, dtype=usetype)
            self.y_array = np.asarray(y_sequences, dtype=usetype)

            stderr.write("Initialized arrays, x: {0} y: {1}\n".format(repr(self.x_array.shape), repr(self.y_array.shape)))

        self.data_len = len(self.x_array)

        # Don't create one-hot encodings by default
        self.x_onehots = None
        self.y_onehots = None
        #self.build_onehots()

    def __getstate__(self):
        state = self.__dict__.copy()

        # References to onehot-encoded shared data
        # shouldn't be serialized here, so remove them
        if 'x_onehots' in state:
            state['x_onehots'] = None

        if 'y_onehots' in state:
            state['y_onehots'] = None

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        if 'data_len' not in state:
            # No data length stored (must be old)
            self.data_len = len(self.x_array)

        if 'charsize' not in state:
            # No vocab size stored (must be old), infer
            stderr.write("No vocabulary size found for onehot conversion, inferring from dataset...\n")
            vocab = np.amax(self.y_array) + 1
            self.charsize = vocab
            stderr.write("Found vocabulary size of: {0:d}\n".format(vocab))
            #self.build_onehots(vocab)

    @staticmethod
    def loadfromfile(filename, charset=None):
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
        Returns filename relative to savedir if successful, None otherwise.
        """
        # Create directory if necessary (won't throw exception if dir already exists)
        #os.makedirs(savedir, exist_ok=True)

        if isinstance(self.srcinfo, str):
            filename = self.srcinfo + ".p"
        elif isinstance(self.srcinfo, dict) and 'name' in self.srcinfo:
            filename = self.srcinfo['name'] + ".p"
        else:
            filename = "dataset.p"

        pathname = os.path.join(savedir, filename)
        try:
            f = open(pathname, 'wb')
        except OSError as e:
            stderr.write("Couldn't open target file {0}, error: {1}\n".format(pathname, e))
            return None
        else:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            stderr.write("Saved data set to {0}\n\n".format(pathname))
            return filename

    def build_onehots(self, vocab_size=None):
        """Build one-hot encodings of each sequence."""

        # If we're passed a charset size, great - if not, fall back to inferring vocab size
        if vocab_size:
            self.charsize = vocab_size
            vocab = vocab_size
        else:
            vocab = self.charsize

        stderr.write("Constructing one-hot vector data...")
        stderr.flush()

        time1 = time.time()

        # These can be large, so we don't necessarily want them on the GPU
        # Thus they're not Theano shared vars
        # Also, numpy fancy indexing is fun!
        self.x_onehots = np.eye(vocab, dtype=th.config.floatX)[self.x_array]
        self.y_onehots = np.eye(vocab, dtype=th.config.floatX)[self.y_array]

        time2 = time.time()

        stderr.write("done!\nTook {0:.4f} ms.\n".format((time2 - time1) * 1000.0))

    def batchepoch(self, batchsize=16):
        """Gets epoch size for given batchsize."""

        # If there's some extra after, we want to extend the batch epoch
        # by 1, so rollover will catch the whole dataset (plus a small 
        # bit of wraparound (easier than padding))
        spacing = self.data_len // batchsize
        offset = 1 if self.data_len % spacing > 0 else 0
        return spacing + offset


    def batch(self, pos=0, batchsize=16):
        """Gets batch of data starting at pos, and evenly spaced along the first
        axis of each onehot array.
        Returns 3-dim ndarrays from x_onehots and y_onehots.
        """
        
        # Find batch spacing and derive indices
        indices = (np.arange(batchsize) * self.batchepoch(batchsize)) + pos

        # Get slices and rearrange
        # Have to transpose so that 2nd/3rd dimensions are matrices corresponding
        # to batchsize rows and onehot columns, and the 1st dim (slice indicies) are
        # the sequences the batch training function will take, since Theano's scan
        # iterates over the first dimension of a given tensor
        if self.x_onehots is not None and self.y_onehots is not None:
            # Onehots already built, take from them
            xbatch = self.x_onehots.take(indices, axis=0, mode='wrap').transpose(1, 0, 2)
            ybatch = self.y_onehots.take(indices, axis=0, mode='wrap').transpose(1, 0, 2)
        else:
            # Build onehots for this batch
            xidxs = self.x_array.take(indices, axis=0, mode='wrap').transpose(1, 0)
            yidxs = self.y_array.take(indices, axis=0, mode='wrap').transpose(1, 0)
            xbatch = np.eye(self.charsize, dtype=th.config.floatX)[xidxs]
            ybatch = np.eye(self.charsize, dtype=th.config.floatX)[yidxs]

        return xbatch, ybatch


class Checkpoint:
    """Checkpoint for model training."""

    def __init__(self, datafile, modelfile, cp_date, epoch, pos, loss, laststate=None):
        self.datafile = datafile
        self.modelfile = modelfile
        self.cp_date = cp_date
        self.epoch = epoch
        self.pos = pos
        self.loss = loss
        self.laststate = laststate

    @classmethod
    def createcheckpoint(cls, savedir, datafile, modelparams, loss, laststate=None):
        """Creates and saves modelparams and pickled training checkpoint into savedir.
        Returns new checkpoint and filename if successful, or (None, None) otherwise.
        """

        # Create directory if necessary (won't throw exception if dir already exists)
        #os.makedirs(savedir, exist_ok=True)

        # Determine filenames
        modeldatetime = datetime.datetime.now(datetime.timezone.utc)
        basefilename = modeldatetime.strftime("%Y-%m-%d-%H:%M:%S-UTC-{0:.3f}-model".format(loss))

        # Save model file
        modelfilename = basefilename + ".npz"
        modelpath = os.path.join(savedir, modelfilename)
        try:
            modelfile = open(modelpath, 'wb')
        except OSError as e:
            stderr.write("Couldn't save model parameters to {0}!\nError: {1}\n".format(modelpath, e))
            return None, None
        else:
            modelparams.savetofile(modelfile)
            modelfile.close()
            stderr.write("Saved model parameters to {0}\n".format(modelpath))

            # Create checkpoint
            cp = cls(datafile, modelfilename, modeldatetime, modelparams.epoch, modelparams.pos, loss, laststate)
            cpfilename = basefilename + ".p".format(loss)
            cppath = os.path.join(savedir, cpfilename)

            # Save checkpoint
            try:
                cpfile = open(cppath, 'wb')
            except OSError as e:
                stderr.write("Couldn't save checkpoint to {0}!\nError: {1}\n".format(cppath, e))
                return None, None
            else:
                pickle.dump(cp, cpfile, protocol=pickle.HIGHEST_PROTOCOL)
                cpfile.close()
                stderr.write("Saved checkpoint to {0}\n".format(cppath))
                return cp, cpfilename

    @classmethod
    def loadcheckpoint(cls, cpfile, fromdir='', fix_old=False):
        """Loads checkpoint from saved file relative to fromdir and returns checkpoint object."""

        cppath = os.path.join(fromdir, cpfile)
        try:
            f = open(cppath, 'rb')
        except OSError as e:
            stderr.write("Couldn't open checkpoint file {0}!\nError: {1}\n".format(cppath, e))
            return None
        else:
            try:
                stderr.write("Restoring checkpoint from file {0}...\n".format(cppath))
                cp = pickle.load(f)
                if fromdir and fix_old:
                    _fix_old_filenames(cp, fromdir)
                # For older versions
                if not hasattr(cp, 'laststate'):
                    cp.laststate = None
            except Exception as e:
                stderr.write("Error restoring checkpoint from file {0}:\n{1}\n".format(cppath, e))
                return None
            else:
                return cp
            finally:
                f.close()

    def printstats(self, outfile):
        """Prints checkpoint stats to file-like object."""

        printstr = """
Checkpoint date: {0}
Dataset file: {1}
Model file: {2}
Epoch: {3:d}
Position: {4:d}
Loss: {5:.4f}
Log loss: {6:.4f}

"""
        outfile.write(printstr.format(
            self.cp_date.strftime("%Y-%m-%d %H:%M:%S %Z"), 
            self.datafile, 
            self.modelfile, 
            self.epoch, 
            self.pos, 
            self.loss,
            log(self.loss)))
        

# TODO: allow initialization from already-constructed charset and dataset
class ModelState:
    """Model state, including hyperparamters, charset, last-loaded 
    checkpoint, dataset, and model parameters.

    Note: Checkpoint is automatically loaded when restoring from file, 
    but dataset and model parameters must be explicitly (re)loaded.
    """
    # Model types
    modeltypes = {
        'GRUResize': GRUResize,
        'GRUEncode': GRUEncode
    }

    def __init__(self, chars, curdir, modeltype='GRUEncode', srcinfo=None, cpfile=None, 
        cp=None, datafile=None, data=None, model=None):
        self.chars = chars
        self.curdir = curdir
        self.modeltype = modeltype
        self.srcinfo = srcinfo
        self.cpfile = cpfile
        self.cp = cp
        self.datafile = datafile
        self.data = data
        self.model = model
        self.laststate = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # References to checkpoint, dataset, and model params 
        # shouldn't be serialized here, so remove them
        state['cp'] = None
        state['data'] = None
        state['model'] = None
        return state

    def __setstate__(self, state):
        # For upgrading of older versions
        if 'modelfile' in state:
            del state['modelfile']
        if 'laststate' not in state:
            state['laststate'] = None

        self.__dict__.update(state)

    @classmethod
    def initfromsrcfile(cls, srcfile, usedir, modeltype='GRUEncode', *, seq_len=100, init_checkpoint=True, **kwargs):
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
            f = open(srcfile, 'r', encoding='utf-8', errors='replace')
        except OSError as e:
            stderr.write("Error opening source file {0}: {1}".format(srcfile, e))
            raise e
        else:
            datastr = f.read()
        finally:
            f.close()

        # Determine full path of working dir and base name of source file
        # Will be using these later on
        # dirname = os.path.abspath(usedir)
        dirname = usedir
        basename = os.path.basename(srcfile)

        # Now find character set
        # TODO: change to let CharSet get chars from string, with frequencies and line beginnings
        charset = CharSet(datastr, srcinfo=(basename + "-chars"))

        # And set hyperparameters (additional keyword args passed through)
        hyperparams = HyperParams(charset.vocab_size, **kwargs)

        # Create dataset, and save
        dataset = DataSet(datastr, charset, seq_len=seq_len, srcinfo=(basename + "-data"))
        datafilename = dataset.savetofile(dirname)

        # Now we can initialize the state
        modelname = "{0:s}-{1:d}x{2:d}-state".format(modeltype, hyperparams.layers, hyperparams.state_size)
        modelstate = cls(charset, dirname, modeltype, srcinfo=modelname, 
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

    @classmethod
    def load(cls, fromdir):
        """Attempts to load model state from file in directory fromdir.
        Will look for files ending in 'state.p'. If multiple files found,
        will print list, but not attempt to load.
        """
        if not fromdir:
            raise FileNotFoundError('No directory specified!')

        # Get list of files from directory
        try:
            filenames = [fn for fn in os.listdir(fromdir) if fn.endswith('state.p')]
        except OSError as e:
            stderr.write("Couldn't list directory, error: {0}\n".format(e))
            return None
        else:
            # No files found
            if len(filenames) == 0:
                stderr.write("No model state file found in {0}\n".format(fromdir))
                return None
            # More than one file found
            elif len(filenames) > 1:
                stderr.write("Multiple model state files found:\n")
                for fn in sorted(filenames, key=str.lower):
                    stderr.write("{0}\n".format(os.path.join(fromdir, fn)))
                return None
            # Just right
            else:
                return cls.loadfromfile(filenames[0], fromdir=fromdir)

    @staticmethod
    def loadfromfile(filename, fromdir='', fix_old=False):
        """Loads model state from filename.
        Note: dataset and model params can be restored from last checkpoint
        after loading model state using restore().
        """

        pathname = os.path.join(fromdir, filename)
        try:
            f = open(pathname, 'rb')
        except OSError as e:
            stderr.write("Couldn't open model state file, error: {0}\n".format(e))
        else:
            try:
                modelstate = pickle.load(f)
            except Exception as e:
                stderr.write("Couldn't load model state, error: {0}\n".format(e))
                return None
            else:
                # Set working directory if specified
                if fromdir:
                    modelstate.curdir = fromdir
                else:
                    modelstate.curdir = os.path.dirname(pathname)
                stderr.write("Using working directory {0}/\n".format(modelstate.curdir))

                # Fix filenames if necessary
                if fromdir and fix_old:
                    _fix_old_filenames(modelstate, fromdir)

                # Reload checkpoint, if present
                if modelstate.cpfile:
                    cppath = os.path.join(modelstate.curdir, modelstate.cpfile)
                    modelstate.cp = Checkpoint.loadcheckpoint(modelstate.cpfile, modelstate.curdir, fix_old)
                    if modelstate.cp:
                        stderr.write("Loaded checkpoint from {0}\n".format(cppath))
                    else:
                        stderr.write("Couldn't load checkpoint from {0}\n".format(cppath))
                        # Checkpoint is invalid, so don't use its file
                        modelstate.cpfile = None
                stderr.write("Loaded model state from {0}\n".format(pathname))

                return modelstate
            finally:
                f.close()

    def savetofile(self, savedir=None):
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
                filename = self.srcinfo + ".p"
            elif isinstance(self.srcinfo, dict) and 'name' in self.srcinfo:
                filename = self.srcinfo['name'] + ".p"
            else:
                filename = "modelstate.p"

            try:
                pathname = os.path.join(usedir, filename)
                f = open(pathname, 'wb')
            except OSError as e:
                stderr.write("Couldn't open target file, error: {0}\n".format(e))
                return None
            else:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
                stderr.write("Saved model state to {0}\n".format(pathname))
                return filename
            finally:
                f.close()

    def loaddata(self, filename=None, fromdir=''):
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
            self.cp = Checkpoint.loadcheckpoint(self.cpfile, self.curdir)
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
        openpath = os.path.join(fromdir, openfile)

        # Load data now that filename is established
        self.data = DataSet.loadfromfile(openpath)
        if self.data:
            self.datafile = openfile
            return True
        else:
            return False

    def loadmodel(self, filename=None, fromdir=''):
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
            self.cp = Checkpoint.loadcheckpoint(self.cpfile, self.curdir)
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
        openpath = os.path.join(fromdir, openfile)

        # Load model now that filename is established
        useclass = self.modeltypes[self.modeltype]
        self.model = useclass.loadfromfile(openpath)
        if self.model:
            return True
        else:
            return False

    def restore(self, cpfile=None, fromdir='', fix_old=False):
        """Restores dataset and model params from specified checkpoint file.
        Defaults to stored checkpoint if none provided.
        """

        if cpfile:
            # Checkpoint given, use that
            cp = Checkpoint.loadcheckpoint(cpfile, fromdir, fix_old)
            if cp:
                self.cp = cp
            else:
                # Try using current directory name at front of given dirname+filename
                cppath = os.path.join(fromdir, cpfile)
                cp = Checkpoint.loadcheckpoint(cppath, self.curdir, fix_old)
                if cp:
                    self.cp = cp
                else:
                    # Still not found (loadcheckpoint will print error for us)
                    return False
        elif self.cp:
            # Try stored checkpoint
            cp = self.cp
        elif self.cpfile:
            # Try loading checkpoint from file
            self.cp = Checkpoint.loadcheckpoint(self.cpfile, self.curdir, fix_old)
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
        if self.loaddata(cp.datafile, self.curdir) and self.loadmodel(cp.modelfile, self.curdir):
            # Load checkpoint's last state, if present
            if isinstance(cp.laststate, np.ndarray):
                self.laststate = cp.laststate
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
        cp, cpfile = Checkpoint.createcheckpoint(usedir, self.datafile, self.model, loss, self.laststate)
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

    def builddataset(self, datastr, seq_len=100, srcinfo=None, savedir=None):
        """Builds new dataset from string and saves to file in working directory."""

        # Build dataset from string
        self.data = DataSet(datastr, self.chars, seq_len, srcinfo)

        # Save to specified dir if provided, otherwise curdir
        usedir = savedir if savedir else self.curdir
        self.datafile = self.data.savetofile(usedir)

        # Return true if both operations succeed
        if self.data and self.datafile:
            return True
        else:
            return False

    def buildmodelparams(self, hyper, checkpointdir=None):
        """Builds model parameters from given hyperparameters and charset size.
        Optionally saves checkpoint immediately after building if path specified.
        """
        useclass = self.modeltypes[self.modeltype]
        self.model = useclass(hyper)

        if checkpointdir:
            # Compile training functions
            self.model._build_t()

            # Get initial loss estimate
            stderr.write("Calculating initial loss estimate...\n")
            
            # We don't need anything fancy or long, just a rough baseline
            data_len = self.data.batchepoch(16)
            loss_len = 50 if data_len >= 50 else data_len
            loss = self.model.calc_loss(self.data, 0, batchsize=8, num_examples=loss_len)

            stderr.write("Initial loss: {0:.3f}\n".format(loss))
            stderr.write("Initial log loss: {0:.3f}\n".format(log(loss)))

            # Take checkpoint
            self.newcheckpoint(loss, savedir=checkpointdir)

    def trainmodel(self, num_rounds=1, batchsize=16, train_len=0, valid_len=0, print_every=1000, clear_state=False):
        """Train loaded model for num_rounds of train_len, printing
        progress every print_every examples, calculating loss using 
        valid_len examples, and creating a checkpoint after each round.

        For train_len and valid_len, a value of 0 indicates using the
        full dataset.

        Validation is performed at the last trained position in the dataset, 
        and training is resumed from the same point after loss is calculated.
        """

        # Make sure we have model and data loaded
        if not self.data or not self.model:
            stderr.write("Dataset and model parameters must be loaded before training!\n")
            return False

        # Compile training functions if not already done
        # First build training functions if not already done
        if not self.model._built_t:
            self.model._build_t()

        # Try block for compatibility with older charsets which haven't done line starts
        try:
            tmpidx = self.chars.semirandomidx()
        except AttributeError:
            self.chars.findlinestarts(self.data.datastr)

        # Progress callback
        progress = printprogress(self.chars)

        # Get max length
        datalen = self.data.batchepoch(batchsize) if batchsize > 0 else self.data.data_len
        train_for = train_len if train_len else datalen
        valid_for = valid_len if valid_len else datalen

        # Start with a blank state if specified, none stored, or wrong shape for batch size
        fresh_state = self.model.freshstate(batchsize)
        if clear_state or not hasattr(self, 'laststate') \
            or not hasattr(self.laststate, 'shape') \
            or self.laststate.shape != fresh_state.shape:
            train_state = fresh_state
        else:
            train_state = self.laststate

        # Print start message
        if batchsize > 0:
            stdout.write(
                "--------\n\nTraining for {0:d} examples with batch size {1:d}, effective epoch length {2:d}\n\n".format(
                train_for * num_rounds, batchsize, datalen))

        # First sample
        if batchsize > 0:
            progress(self.model, train_state[:,0,:])
        else:
            progress(self.model, train_state)

        time1 = time.time()

        # Train for num_rounds
        for roundnum in range(num_rounds):
            # Train...
            train_state = self.model.train(
                self.data,
                batchsize=batchsize,
                num_examples=train_for,
                callback=progress,
                callback_every=print_every,
                init_state=train_state)

            # Calc loss
            stdout.write("--------\n\nCalculating loss (epoch {0:d}, pos {1:d})...\n".format(
                self.model.epoch, self.model.pos))
            stdout.flush()

            # Calculate loss with blank state
            #loss = self.model.calc_loss(x_slice, y_slice)
            loss = self.model.calc_loss(self.data, self.model.pos, batchsize=batchsize, num_examples=valid_len)

            stdout.write("Previous loss: {0:.4f}, current loss: {1:.4f}\n".format(self.cp.loss, loss))
            stdout.write("Previous log loss: {0:.4f}, current log loss: {1:.4f}\n".format(log(self.cp.loss), log(loss)))

            # Adjust learning rate if necessary
            lossdiff = log(loss) / log(self.cp.loss)
            if lossdiff >= 1.03:
                # Loss increasing, lower learning rate
                self.model.hyper.learnrate *= 0.5
                stdout.write("Loss increased between validations, adjusted learning rate to {0:.6f}\n".format(
                    self.model.hyper.learnrate))
            elif lossdiff < 1.03 and lossdiff > 0.98:
                # Loss not decreasing fast enough, lower learning rate a bit
                self.model.hyper.learnrate *= 0.8
                stdout.write("Loss changed too little between validations, adjusted learning rate to {0:.6f}\n".format(
                    self.model.hyper.learnrate))

            stdout.write("\n--------\n\n")

            # Save training state for later
            # With batched training, unless one trains for epochs equal to batch size,
            # no one state vector will see the entire dataset, so we save between runs
            # to keep continuity
            self.laststate = train_state

            # Take checkpoint and print stats
            self.newcheckpoint(loss)
            self.cp.printstats(stdout)

        time2 = time.time()
        timetaken = time2 - time1

        stdout.write("Completed {0:d} rounds of {1:d} examples each.\n".format(num_rounds, train_for))
        stdout.write("Total time: {0:.3f}s ({1:.3f}s per round).\n".format(timetaken, timetaken / float(num_rounds)))

    def generatestring(self, numchars=100, temp=0.5, init_state=None, ret_state=False):
        '''Generate string from current model state.'''
        genstr, newstate = self.model.genchars(self.chars, numchars, 
            init_state=init_state, temperature=temp)
        print("--------\nGenerated {0} chars, temperature {1}\n--------\n\n{2}\n".format(
            numchars, temp, genstr))
        if ret_state:
            return newstate
        else:
            return None

    def trackneurons(self, usestr, temp=0.5, ret_output=False):
        '''Track output of model for given string input. Useful for observing neuron
        activity as a sequence is processed.
        Returns 3D array of states along sequence, in shape (layer, neuron, sequence).
        Optionally returns prediction outputs if ret_output=True.
        '''
        # Encode string as onehots
        onehots = self.chars.encodeonehots(usestr)
        # Get new state
        start_state = self.model.freshstate(0)
        # Process sequence
        out_seq, state_seq = self.model.seq_process(onehots, start_state, temp)
        # Reshuffle state output
        state_t = state_seq.transpose(1, 2, 0)
        if ret_output:
            return out_seq, state_t
        else:
            return state_t


# Unattached functions

def printprogress(charset):
    def retfunc (model, init_state=None):
        print("--------\n")
        print("Time: {0}".format(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")))
        print("Epoch: {0}, pos: {1}".format(model.epoch, model.pos))
        print("Generated 100 chars:\n")
        genstr, _ = model.genchars(charset, 100, init_state=init_state, temperature=0.5)
        print(genstr + "\n")
    return retfunc

def _fix_old_filenames(obj, fromdir):
    """Rewrite stored filenames from old versions (including paths) to
    new versions (relative to dir).
    """
    # Internal func to get proper filename
    def _fixedname(filename, fromdir):
        if os.path.isabs(filename):
            return filename
        else:
            basename = os.path.basename(filename)
            if os.path.join(fromdir, basename) == filename:
                return basename
            elif os.path.dirname(filename):
                return os.path.relpath(filename, fromdir)
            else:
                return filename

    # Check for various filename attributes on obj
    if hasattr(obj, 'cpfile'):
        obj.cpfile = _fixedname(obj.cpfile, fromdir)
    if hasattr(obj, 'datafile'):
        obj.datafile = _fixedname(obj.datafile, fromdir)
    if hasattr(obj, 'modelfile'):
        obj.modelfile = _fixedname(obj.modelfile, fromdir)


# TODO: Non-Theano sigmoid
# TODO: Non-Theano softmax
# TODO: Command-line options for generation, training

