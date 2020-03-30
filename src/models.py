import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import Sequence
import os
import json
import time


def build_dialect_classification_model(lstm_output_dim=128,
                                       encoding_dimensions=247,
                                       dropout_rate=0.3,
                                       output_activation='tanh',
                                       loss='categorical_crossentropy',
                                       random_seed=2020):
    """
    Creates and compiles a sequential model for dialect classification.

    Parameters
    ----------
    lstm_output_dim:integer, optional
        The number of dimensions of the LSTM output.
        Default is 128.
    encoding_dimensions: integer, optional
        The size of the array representing a single character.
        Default is 247.
    dropout_rate: float, optional
        The fraction of the LSTM outputs to drop. Default is 0.3.
    output_activation: string, optional
        The activation function of the output layer.
        Default is 'tanh'.
    loss: string, optional
        The loss function. Default is 'categorical_crossentropy'.
    random_seed:integer, optional
        The seed for random number generator. Default is 2020.
    """
    model = Sequential()
    model.add(LSTM(lstm_output_dim, input_shape=(None, encoding_dimensions)))
    model.add(Dropout(dropout_rate, seed=random_seed))
    model.add(Dense(2, activation=output_activation))
    optimizer = Adam()
    model.compile(optimizer, loss=loss, metrics=['accuracy'])
    return model


def test_model(model,
               tokenizer,
               samples,
               labels,
               num_predictions=None,
               shuffle=True):
    """
    Tests model predictions.

    Parameters
    ----------
    model: keras.models.Sequential
        The dialect classification model.
    tokenizer: keras.preprocessing.text.Tokenizer
        The tokenizer used to transform sample to numerical representation.
    samples: list of string
        The samples to use for testing.
    labels: list of integer
        The associated labels of the samples.
    num_predictions: integer, optional
        Restrict test to `num_predictions`.
        Default is `None` which means test all samples.
    shuffle: boolean, optional
        Specifies whether to shuffle the samples and labels before testing.
        Default is `True`.
    """
    if num_predictions is None:
        num_predictions = len(samples)
    else:
        num_predictions = min(num_predictions, len(samples))

    if shuffle:
        indices = np.random.permutation(len(samples))
    else:
        indices = np.arange(len(samples))

    indices = indices[:num_predictions]
    for idx in indices:
        sample = samples[idx]
        label = labels[idx]
        x = tokenizer.texts_to_matrix(sample)
        x = np.reshape(x, (1, *x.shape))
        y = model.predict(x)
        pred = np.argmax(y) + 1  # class labels are 1,2 not 0,1
        print("Label: {}, prediction: {}".format(label, pred))


def save_model(model, tokenizer, output_path):
    """
    Saves the model and tokenizer to te specified path.

    Parameters
    ----------
    model: keras.models.Sequential
        The dialect classification model.
    tokenizer: keras.preprocessing.text.Tokenizer
        The text tokenizer.
    output_path: string
        The directory where to save model and tokenizer.
    """
    file_suffix = time.strftime("%Y%m%d%H%M%S")
    file_name = 'dialect-classifier-model-{}.h5'.format(file_suffix)
    model.save(os.path.join(output_path, file_name))

    file_name = 'tokenizer-{}.json'.format(file_suffix)
    tokenizer_json = tokenizer.to_json()
    file_name = os.path.join(output_path, file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


class SingleSampleBatchGenerator(Sequence):
    """
    Generates batches containing a single sample for Keras models.
    Adapted from https://datascience.stackexchange.com/a/48814/45850

    """
    def __init__(self, samples, labels):
        """
        Initalizes an instance of SingleSampleBatchGenerator.

        Parameters
        ----------
        samples: list of numpy.ndarray
            The samples to be batched.
        labels: list of numpy.ndarray
            The labels associated with the samples.
        """
        super(SingleSampleBatchGenerator, self).__init__()
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
        samples_batch = np.empty((1, *self.samples[index].shape))
        labels_batch = np.empty((1, *self.labels[index].shape))

        samples_batch[0] = self.samples[index]
        labels_batch[0] = self.labels[index]
        return samples_batch, labels_batch
