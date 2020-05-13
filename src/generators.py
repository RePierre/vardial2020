import numpy as np
from keras.utils import Sequence


class SingleSampleBatchGenerator(Sequence):
    """
    Generates batches containing a single sample for Keras models.
    Adapted from https://datascience.stackexchange.com/a/48814/45850

    """
    def __init__(self, tokenizer, samples, labels):
        """
        Initalizes an instance of SingleSampleBatchGenerator.

        Parameters
        ----------
        tokenizer: keras.preprocessing.text.Tokenizer
            The trained tokenizer used to transform a sample into a matrix.
        samples: list of text
            The samples to be batched.
        labels: list integers
            The labels associated with the samples.
        """
        super(SingleSampleBatchGenerator, self).__init__()
        self.tokenizer = tokenizer
        self.samples = samples
        self.labels = labels

    def __len__(self):
        """
        Returns number of batches per epoch.
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Returns the batch at the specified index.
        """
        x = self.tokenizer.texts_to_matrix(self.samples[index])
        y = np.zeros((1, 2))
        y[0, self.labels[index] - 1] = 1

        sample_shape = x.shape
        x = np.reshape(x, (1, *sample_shape))

        return x, y


class SequencePaddingBatchGenerator(Sequence):
    """
    Generates batches of input samples.
    Adapted from https://datascience.stackexchange.com/a/48814/45850
    """
    def __init__(self,
                 tokenizer,
                 samples,
                 labels,
                 max_sequence_length,
                 batch_size=32,
                 mask_value=-13):
        """
        Initializes a new instance of SequencePaddingBatchGenerator.

        Parameters
        ----------
        tokenizer: keras.preprocessing.text.Tokenizer
            The trained tokenizer used to transform a sample into a matrix.
        samples: list of text
            The samples to be batched.
        labels: list integers
            The labels associated with the samples.
        max_sequence_length: integer
            The maximum length of a sequence.
        batch_size: integer, optional
            The size of each batch. Default is 32.
        mask_value: integer
            The value used for masking padded tensors.
        """
        super(SequencePaddingBatchGenerator, self).__init__()
        self.tokenizer = tokenizer
        self.samples = samples
        self.labels = labels
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.mask_value = mask_value
        self.indexes = np.arange(len(self.labels))
        self.on_epoch_end()
        self.sample_dim = self.get_datapoint_dim()

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        """
        Returns the batch at the specified index.
        """
        batch_start = index * self.batch_size
        batch_end = batch_start + self.batch_size - 1
        samples = [
            self.samples[self.indexes[i]]
            for i in range(batch_start, batch_end)
        ]
        labels = [
            self.labels[self.indexes[i]]
            for i in range(batch_start, batch_end)
        ]
        x = np.full(
            (self.batch_size, self.max_sequence_length, self.sample_dim),
            self.mask_value)
        for i, s in enumerate(samples):
            sample = self.tokenizer.texts_to_matrix(s)
            sample_len = sample.shape[0]
            x[i, 0:sample_len, :] = sample
        y = np.zeros((self.batch_size, 2))
        for i, l in enumerate(labels):
            y[i, l - 1] = 1

        return x, y

    def on_epoch_end(self):
        """
        Shuffle data indices after each epoch.
        """
        np.random.shuffle(self.indexes)

    def get_datapoint_dim(self):
        """
        Returns the number of dimensions for each data point in the sample
        """
        x = self.tokenizer.texts_to_matrix(self.samples[0])
        return x.shape[1]
