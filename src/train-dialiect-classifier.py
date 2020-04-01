import logging
from argparse import ArgumentParser

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from input import VarDialDataSet
from models import build_dialect_classification_model
from models import SingleSampleBatchGenerator, save_model, test_model


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

    logging.info('Training tokenizer on text...')
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(samples_train)

    logging.info('Building model...')
    encoding_dim = tokenizer.texts_to_matrix(samples_train[0]).shape[1]
    model = build_dialect_classification_model(args.lstm_output_dim,
                                               encoding_dim, args.dropout_rate)
    print(model.summary())
    logging.info('Training the model...')
    train_batch_generator = SingleSampleBatchGenerator(tokenizer,
                                                       samples_train,
                                                       labels_train)
    model.fit_generator(train_batch_generator,
                        epochs=args.num_epochs,
                        workers=args.num_train_workers,
                        use_multiprocessing=args.use_multiprocessing)

    logging.info('Scoring the model...')
    eval_batch_generator = SingleSampleBatchGenerator(tokenizer, samples_test,
                                                      labels_test)
    score, acc = model.evaluate_generator(eval_batch_generator)
    print('Test score: {}'.format(score))
    print('Test accuracy: {}.'.format(acc))
    test_model(model,
               tokenizer,
               samples_test,
               labels_test,
               num_predictions=args.num_predictions,
               shuffle=not args.no_shuffle_before_predict)
    logging.info('Saving model and tokenizer...')
    save_model(model, tokenizer, args.save_model_to)
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
    parser.add_argument('--lstm-output-dim',
                        help='The number of dimensions of the LSTM output.',
                        type=int,
                        default=128)
    parser.add_argument('--dropout-rate',
                        help='Fraction of the LSTM outputs to drop.',
                        type=float,
                        default=0.3)
    parser.add_argument('--num-epochs',
                        help='Number of training epochs.',
                        type=int,
                        default=10)
    parser.add_argument('--num-predictions',
                        help='Number of predictions to make on test data.',
                        type=int,
                        default=15)
    parser.add_argument(
        '--no-shuffle-before-predict',
        help='Do not shuffle the data before making predictions.',
        action='store_false')
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
    parser.add_argument('--num-train-workers',
                        help='The number of workers to use for training.',
                        type=int,
                        default=1)
    parser.add_argument(
        '--use-multiprocessing',
        help='Specifies whether to use or not multiprocessing when training.',
        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    args = parse_arguments()
    run(args)
