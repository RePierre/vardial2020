import logging
import numpy as np
from argparse import ArgumentParser
from models import load_model
from utils import reshape_input_data
from constants import DialectLabels


def load_data(file_name):
    """
    Loads the test samples from the specified file.

    Parameters
    ----------
    file_name: string
        The path to the file containing test data.
    """
    with open(file_name, 'rt') as f:
        samples = f.readlines()

    return [s.strip() for s in samples]


def save_predictions(labels, output_file):
    """
    Save the predicted labels to the specified file.

    Parameters
    ----------
    labels: iterable of integer
        The iterable containing predicted labels.
    output_file: string
        The name of the file where to save the labels.
    """
    with open(output_file, 'wt') as f:
        for l in labels:
            f.write("{}\n".format(l))


def run(args):
    logging.info("Loading data...")
    samples = load_data(args.test_file)

    logging.info("Loading model...")
    model, ro_vectorizer, md_vectorizer = load_model(args.model_path,
                                                     args.ro_vectorizer_path,
                                                     args.md_vectorizer_path)

    logging.info("Start predicting.")
    label_map = {DialectLabels.Moldavian: "MD", DialectLabels.Romanian: "RO"}
    labels = []
    for sample in samples:
        x = reshape_input_data(ro_vectorizer.transform([sample]),
                               md_vectorizer.transform([sample]))
        y = model.predict(x)
        label = np.argmax(y[0, :]) + 1
        if not args.use_numeric_labels:
            label = label_map[label]
        labels.append(label)
        logging.info("Predicted label [{}] for sample [{}].".format(
            label, sample))

    logging.info("Saving predictions to {}...".format(args.output_file))
    save_predictions(labels, args.output_file)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model-path', help='The path to the saved model.')
    parser.add_argument('--ro-vectorizer-path',
                        help='The path to the pickled Romanian vectorizer.')
    parser.add_argument('--md-vectorizer-path',
                        help='The path to the pickled Moldavian vectorizer.')
    parser.add_argument('--test-file',
                        help='The path to the file containing test data.',
                        default='test.txt')
    parser.add_argument('--output-file',
                        help='The file where to save predicted labels.',
                        default='labels.txt')
    parser.add_argument(
        '--use-numeric-labels',
        help="If specified, the output file will contain numeric labels.",
        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    args = parse_arguments()
    run(args)
    logging.info("That's all folks!")
