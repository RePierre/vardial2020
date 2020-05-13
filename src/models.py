import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import GlobalMaxPool1D
from keras.layers import Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import Sequence
from scipy.sparse import csr_matrix
import os
import time
import pickle


def build_dialect_classification_model(input_shape,
                                       dropout_rate=0.3,
                                       loss='categorical_crossentropy',
                                       random_seed=2020):
    """
    Creates and compiles a sequential model for dialect classification.

    Parameters
    ----------
    input_shape: tuple
        The shape of the input data.
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
    model.add(
        Conv1D(filters=256,
               kernel_size=3,
               padding='valid',
               activation='relu',
               strides=1,
               input_shape=input_shape))
    model.add(GlobalMaxPool1D())
    model.add(Dense(128))
    model.add(Dropout(dropout_rate, seed=random_seed))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    optimizer = Adam()
    model.compile(optimizer, loss=loss, metrics=['accuracy'])
    return model


def reshape_input_data(x_ro, x_md):
    """
    Concatenates the input data into shape (num_samples, sample_size, 2).


    Parameters
    ----------
    x_ro: sparse matrix
        TF-IDF encoding of Romanian input samples.
    x_md: sparse matrix
        TF-IDF encoding of Moldavian input samples.

    Returns
    -------
    result
        Numpy ndarray representing the concatenated data.
    """
    assert x_ro.shape == x_md.shape
    num_samples, sample_size = x_ro.shape
    result = np.stack([csr_matrix.toarray(x_ro),
                       csr_matrix.toarray(x_md)],
                      axis=-1)
    return result


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


def save_model(model, ro_vectorizer, md_vectorizer, output_path):
    """
    Saves the model and TF-IDF vectorizers to te specified path.

    Parameters
    ----------
    model: keras.models.Sequential
        The dialect classification model.
    ro_vectorizer: sklearn TfIdfVectorizer
        The vectorizer trained on Romanian dialect.
    md_vectorizer: sklearn TfIdfVectorizer
        The vectorizer trained on Moldavian dialect.
    output_path: string
        The directory where to save model and tokenizer.
    """
    file_suffix = time.strftime("%Y%m%d%H%M%S")
    file_name = 'dialect-classifier-model-{}.h5'.format(file_suffix)
    model.save(os.path.join(output_path, file_name))

    def save_vectorizer(vectorizer, vectorizer_type):
        file_name = '{}-vectorizer-{}.pkl'.format(vectorizer_type, file_suffix)
        file_name = os.path.join(output_path, file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(vectorizer, f)

    save_vectorizer(ro_vectorizer, 'ro')
    save_vectorizer(md_vectorizer, 'md')


def get_max_sequence_length(samples, coefficient=0.2):
    """
    Returns the maximum length of sample sequences plus a buffer.

    Parameters
    ----------
    samples: list of strings
        The training samples.
    coefficient: float
        The coefficient by which to multiply the maximum
        sequence length determined from data.

    Returns
    -------
    max_len: integer
        The maximum sequence length.
    """
    max_len = max([len(s) for s in samples])
    return int(np.ceil(max_len + max_len * coefficient))


def encode_dialect_labels(dialect_labels):
    """
    Encodes the dialect labels into one-hot representation.

    Parameters
    ----------
    dialect_labels: iterable of integer
        The dialect labels.

    Returns:
    labels_encoded
        The encoded dialect labels.
    """
    y = np.zeros((len(dialect_labels), 2))
    for i, l in enumerate(dialect_labels):
        y[i, l - 1] = 1

    return y
