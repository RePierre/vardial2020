import numpy as np
from scipy.sparse import csr_matrix


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
