import logging
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from input import VarDialDataSet
from models import build_dialect_classification_model
from models import reshape_input_data
from models import encode_dialect_labels
from models import save_model
from feature_extraction import train_dialect_vectorizers
from feature_extraction import build_common_vocabulary


def run(args):
    logging.info('Loading data...')
    ds = VarDialDataSet(args.data_directory,
                        samples_file=args.samples_file,
                        dialect_labels_file=args.dialect_labels_file,
                        category_labels_file=args.category_labels_file)
    ds.initialize()
    samples = ds.samples
    labels = ds.dialect_labels

    if args.debug:
        logging.info(
            'Running in debug mode. Dataset is restricted to {} samples.'.
            format(args.num_debug_samples))
        samples = ds.samples[:args.num_debug_samples]
        labels = ds.dialect_labels[:args.num_debug_samples]

    samples_train, samples_test, \
        labels_train, labels_test = train_test_split(samples, labels)

    logging.info('Training vectorizers on text...')
    vocab = build_common_vocabulary(samples_train)
    ro, md = train_dialect_vectorizers(samples_train, labels_train, vocab)
    logging.info('Reshaping input data...')
    x = reshape_input_data(ro.transform(samples_train),
                           md.transform(samples_train))
    print(x.shape)

    logging.info('Building model...')
    model = build_dialect_classification_model(x.shape[1:], args.dropout_rate)
    print(model.summary())

    logging.info('Training the model...')
    y = encode_dialect_labels(labels_train)
    print(y.shape)
    model.fit(x=x, y=y, batch_size=args.batch_size, epochs=args.num_epochs)

    logging.info('Scoring the model...')
    x = reshape_input_data(ro.transform(samples_test),
                           md.transform(samples_test))
    y = encode_dialect_labels(labels_test)
    score, acc = model.evaluate(x=x, y=y, batch_size=args.batch_size)
    print('Test score: {}'.format(score))
    print('Test accuracy: {}.'.format(acc))

    logging.info('Saving model and vectorizers...')
    save_model(model, ro, md, args.save_model_to)
    logging.info("That's all folks!")


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data-directory',
                        help='Path to the directory containing training data.',
                        default='/data')
    parser.add_argument('--samples-file',
                        help='The name of the file containing text samples.',
                        default='samples.txt')
    parser.add_argument('--dialect-labels-file',
                        help='The name of the file containing dialect labels.',
                        default='dialect_labels.txt')
    parser.add_argument('--category-labels-file',
                        help='The file containing category labels.',
                        default='category_labels.txt')
    parser.add_argument('--dropout-rate',
                        help='Dropout rate.',
                        type=float,
                        default=0.2)
    parser.add_argument('--num-epochs',
                        help='Number of training epochs.',
                        type=int,
                        default=10)
    parser.add_argument('--num-predictions',
                        help='Number of predictions to make on test data.',
                        type=int,
                        default=15)
    parser.add_argument('--save-model-to',
                        help='Path to the directory where to save the model.',
                        default='.')
    parser.add_argument('--debug',
                        help='Signals that the script is in debug mode.',
                        action='store_true')
    parser.add_argument(
        '--num-debug-samples',
        help="Specifies the number of samples to use for debugging.",
        type=int,
        default=100)
    parser.add_argument('--batch-size',
                        help="Batch size for training.",
                        type=int,
                        default=16)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    args = parse_arguments()
    run(args)
