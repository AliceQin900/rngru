# RN-GRU
This implements a multi-layer gated recurrent unit neural network project in Python/Theano, for training and sampling from character-level models. The code is inspired by Andrej Karpathy's (@karpathy) [char-rnn](https://github.com/karpathy/char-rnn) and Denny Britz' (@dennybritz) [WildML RNN tutorial](https://github.com/dennybritz/rnn-tutorial-gru-lstm).

The model will encode an input source text file as sequences of characters, and trains a GRU neural network to predict the next character in a sequence. The model can then be sampled at various temperatures to generate text. Training is batched by default, with user-specified batch size (default 16). Input text is split into fixed-size sequences (the last sequence will be padded with text from the beginning of the file if necessary) for ease of batching. Training round durations can be partial epochs, for testing initial hyperparameter choices. Short generated samples are printed during training.

Hidden states are stored between sequences, as the input file is not split by line or delimiter, thus each training sequence is presented to the model as a continuous stream with regards to the hidden state. At epoch rollover, the hidden state is shuffled so that each part of the batch advances through the source continuously. The position of the model within the dataset is stored, along with the last returned hidden state, so that training for partial epochs doesn't affect continuity of the hidden states of each layer. Batches are constructed from evenly-spaced slices of the dataset, to prevent overlap and maintain continuity for each segment in a batch.

A character set is constructed from the input file, and data is encoded as one-hot vectors of length equal to the charset size. (As an implementation detail, sequences are stored as integer indices, and translated to one-hot vectors on demand.) At present, due to its primary function as a text generator, validation is performed simply using the next section of training data (thus the least recently seen by the model) instead of partitioning the data into training and validation sets -- I'll likely be changing this in the near future.

Two variants are (currently) available, GRUResize and GRUEncode. GRUResize uses the first recurrent layer as the input layer, taking in a vocabulary-sized vector and giving a hidden-state-sized vector as output. Additional layers input and output state-sized vectors, and then an output layer translates from state-sized vector to vocab-sized vector. GRUEncode uses a separate non-recurrent layer (using tanh as the activation function) as the input layer, and all hidden layers input and output state-sized vectors.

Personally, I've had more luck with GRUEncode, so that's default.

## Requirements
The code is written in Python (with Numpy), and uses [Theano](http://deeplearning.net/software/theano/) to provide auto-differentiation and GPU acceleration (GPU use requires additional configuration -- see the [Theano documentation](http://deeplearning.net/software/theano/tutorial/using_gpu.html) for details). Minimum Python version is 3.4 and minimum Theano version is 0.7, along with any additional dependencies from these.

## Usage
As Theano's compliation step can be time-consuming, this code is presently designed to be used in the interactive Python shell. To launch, go to the project directory and use:
```
python3 -i rn_rnn_char.py
```
Once at the shell, use ```help(ModelState)``` and ```help(HyperParams)``` to familiarize yourself with the functions available.

All the main functions regarding initialization, training, and generation are accessible through the ModelState class. To initialize a model from a text file, use a command similar to:
```
ms = ModelState.initfromsrcfile('/path/to/txt/file.txt', 'workingdir', modeltype='GRUEncode', layers=3, state_size=256, learnrate=0.0005, decay=0.9, regcost=0.2)
```
By default, the parameter matrices will be initialized, the untrained loss will be calculated, and a starting checkpoint will be created.

To train the model:
```
ms.trainmodel(num_rounds=3, batchsize=16, valid_len=200, print_every=100)
```

To generate text:
```
generatestring(ms, numchars=3000, temp=0.5)
```

The model state, encoded dataset, and checkpoint files are stored in the working directory specified in initfromsrcfile(). To reload a saved model, use:
```
ms = ModelState.load('workingdir')
```
This will reload the model state from the last checkpoint. However, since Theano compilation can be slow, the model paramters and text generation functions are not automatically reloaded. To do so, use:
```
ms.restore()
```
This will load the model's parameter matrices and recompile text generation functions. An additional training function compilation step will be automatically performed if trainmodel() is called. To restore from a previous checkpoint:
```
ms.restore('checkpoint-filename')
```

## License
Apache 2.0 (see LICENSE.txt for details)
