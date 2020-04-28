import logging
from constants import DialectLabels
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def build_and_train_vectorizer(corpus, vocabulary):
    """
    Builds and trains an instance of TfidfVectorizer.

    Parameters
    ----------
    corpus: list of str
        The training corpus.
    vocabulary: iterable of str
        The vocabulary to train vectorizers on.
    """
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    vectorizer.fit(corpus)
    return vectorizer


def get_samples_for_dialect(samples, labels, dialect):
    """
    Returns the samples for the specified dialect.

    Parameters
    ----------
    samples: iterable of str
        The samples to be filtered.
    labels: iterable of integer
        The dialect labels of the samples.
    dialect: integer
        The dialect for which to return samples.

    Returns
    -------
    dialect_samples
        A list containing samples for the specified dialect.
    """
    indices = [idx for idx, label in enumerate(labels) if label == dialect]
    return [samples[idx] for idx in indices]


def train_dialect_vectorizers(samples,
                              dialect_labels,
                              vocabulary,
                              max_features=1000):
    """
    Builds and trains TF-IDF vectorizers for each dialect.

    Parameters
    ----------
    samples: iterable of str
        The samples on which to train vectorizers.
    dialect_labels: iterable of integer
        The dialect labels of the samples.
    vocabulary: iterable of str
        The vocabulary to train vectorizers on.
    max_features: integer, optional
        The number of features to consider when training the vectorizer.
        Default is 1000.

    Returns
    -------
    (ro_vectorizer, md_vectorizer)
        A tuple of TfidfVectorizer instances, one for each of the two dialects.
    """

    logging.info("Training TF-IDF vectorizer on Romanian dialect...")
    dialect_samples = get_samples_for_dialect(samples, dialect_labels,
                                              DialectLabels.Romanian)
    ro_vectorizer = build_and_train_vectorizer(dialect_samples, vocabulary)
    logging.info("Training TF-IDF vectorizer on Moldavian dialect...")
    dialect_samples = get_samples_for_dialect(samples, dialect_labels,
                                              DialectLabels.Moldavian)
    md_vectorizer = build_and_train_vectorizer(dialect_samples, vocabulary)
    logging.info("Done.")
    return ro_vectorizer, md_vectorizer


def build_common_vocabulary(samples, named_entity_token=None):
    """
    Builds a common vocabuldary to be used with TF-IDF
    vectorizers for each language.

    Parameters
    ----------
    samples: iterable of str, required
        The samples from which to build vocabuldary.
    named_entity_token: str, optional
        Represents the token used to replace a named entity.
        If provided, the value will be excluded from result dictionary.
        Default is None.

    Returns
    -------
    vocabuldary
        An iterable of strings representing the words in the vocabuldary.
    """
    vectorizer = CountVectorizer()
    vectorizer.fit(samples)
    return [
        feature for feature in vectorizer.get_feature_names()
        if feature != named_entity_token
    ]


if __name__ == "__main__":
    from input import VarDialDataSet

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    ds = VarDialDataSet('/data')
    ds.initialize()
    ds.preview()

    samples = ds.samples[:10]
    dialect_labels = ds.dialect_labels[:10]

    features = build_common_vocabulary(samples)
    print("Common vocabuldary:")
    print(features)
    ro, md = train_dialect_vectorizers(samples, dialect_labels, features)

    print("Testing vectorizers.")
    print('Romanian representation:')
    x = ro.transform(samples)
    print(x)
    print("Shape: {}.".format(x.shape))

    print('Moldavian representation:')
    x = md.transform(samples)
    print(x)
    print("Shape: {}.".format(x.shape))
