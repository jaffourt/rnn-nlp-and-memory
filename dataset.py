from pathlib import Path
from torchnlp.datasets import penn_treebank_dataset
from torchnlp.encoders.text import DEFAULT_EOS_TOKEN
from torch import zeros, tensor
import numpy as np


class Dataset:

    # TODO: Make this compatible with neurogym

    def __init__(self, **kwargs):
        self._dataset = penn_treebank_dataset
        self._eos_token = DEFAULT_EOS_TOKEN
        self._train_testing_split = 0.2
        self._i_batch = 0
        self._seq_start = 0
        self._inputs = None
        self._targets = None
        self._seq_end = None

        self.data = {}
        self.vocabulary = None

        self.embedding_size = kwargs.get('embedding_size', 32)  # TODO: infer this from the data
        self.batch_size = kwargs.get('batch_size', 16)
        self.max_batch = kwargs.get('max_batch', np.inf)
        self.word_level = kwargs.get('word_level', True)
        self.batch_first = kwargs.get('batch_first', False)

        self.init(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.__next__(**kwargs)

    def __iter__(self):
        return self

    def __next__(self, **kwargs):
        self._i_batch += 1

        if self._i_batch > self.max_batch:
            self._i_batch = 1
            self._seq_start = 0
            self._seq_end = self._seq_start + self.embedding_size
            # raise StopIteration  # TODO: finished the first epoch, save state / eval?

        self._seq_end = self._seq_start + self.embedding_size

        if self.batch_first:
            inputs = self._inputs[:, self._seq_start:self._seq_end]
            target = self._targets[:, self._seq_start:self._seq_end]
        else:
            inputs = self._inputs[self._seq_start:self._seq_end]
            target = self._targets[self._seq_start:self._seq_end]

        self._seq_start = self._seq_end
        return inputs, target

    def init(self, **kwargs):
        # download first, to avoid reliance on internet connection
        if not Path('data/penn-treebank/').exists():
            self._dataset()

        self.get_dataset()
        if self.word_level:
            self.vocabulary = Vocabulary(data=[*self.data['train']],  # *self.data['valid'], *self.data['test']
                                         max_items=self.batch_size * self.embedding_size * self.max_batch,
                                         default_indexes={0: self._eos_token, 1: '<unk>', 2: 'N'})
        else:
            # basic preprocessing for character level
            data = ' '.join(self.data['train'])
            data = [sentence for sentence in data.split(' </s>')]
            data = '¬'.join(data)  # the end sentence character..
            data = data.replace('N', '#')  # the number character..
            data = data.replace('<unk>', '?')  # the unknown word character..
            self.data['train'] = list(data)
            self.vocabulary = Vocabulary(data=self.data['train'], max_items=self.max_batch * self.embedding_size)

        self._inputs, self._targets = self.embed_dataset()

        self._seq_start = 0
        self._seq_end = self._seq_start + self.embedding_size

    def get_dataset(self, **kwargs):
        self.data['train'] = self._dataset(train=True, eos_token=self._eos_token)
        self.data['valid'] = self._dataset(dev=True, eos_token=self._eos_token)
        self.data['test'] = self._dataset(test=True, eos_token=self._eos_token)

    def embed_dataset(self):
        # word based
        if self.word_level:
            inputs = [self.vocabulary.item_to_index[word] for word in self.data['train'][: self.vocabulary.max_items]]
            targets = inputs[1:]
            targets.append(inputs[0])
            return np.reshape(inputs, (self.batch_size, -1)), np.reshape(targets, (self.batch_size, -1))
        # character based
        else:
            # TODO: create tensor each __iter to save time
            inputs = [self.vocabulary.item_to_index[char] for char in self.data['train'][: self.vocabulary.max_items]]
            targets = inputs[1:]
            targets.append(inputs[0])
            return self.vocabulary.one_hot(inputs), tensor(targets)


class Vocabulary:
    """ Vocabulary for RNN language models

        Args:
            data: all of the words in a single list
            default_indexes: dictionary of default indices

    """

    def __init__(self, data=None, max_items=0, default_indexes=None):

        # hidden for now
        self._word_counts = {}

        # mutable default arguments
        if default_indexes is None:
            default_indexes = {}
        if data is None:
            data = []

        self.vocab_size = 0

        # TODO: max_words gathered from train only, validate and test needs to be added
        self.max_items = max_items
        self.item_to_index = {}
        self.index_to_item = {**default_indexes}

        # TODO: validate that unpack maintains order or combine the sets seperately
        self.data = data[:max_items]
        self.init()

    def init(self):

        for ind, word in self.index_to_item.items():
            self.item_to_index[word] = ind

        self.counter()
        self.embedding()

    def counter(self):
        for word in self.data:
            if word not in self._word_counts:
                self._word_counts[word] = 1
            else:
                self._word_counts[word] += 1
        self.vocab_size = len(self._word_counts)

    def embedding(self):
        for word in sorted(self._word_counts):
            if word not in self.item_to_index:
                self.item_to_index[word] = len(self.item_to_index)
                self.index_to_item[len(self.index_to_item)] = word

    def one_hot(self, line):
        encoded = zeros(len(line), 1, self.vocab_size)
        for i in range(encoded.shape[0]):
            encoded[i][0][line[i]] = 1
        return encoded
