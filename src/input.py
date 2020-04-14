import logging
import os.path as path
import pandas as pd


class VarDialDataSet(object):
    """
    Represents the data set for VarDial 2020 Dialect Identification Task

    """
    def __init__(self,
                 data_dir,
                 samples_file='samples.txt',
                 dialect_labels_file='dialect_labels.txt',
                 category_labels_file='category_labels.txt'):
        """
        Initalizes a new instance of VarDialDataSet class.

        Parameters
        ----------
        data_dir: string
            The path to the directory containing the data.
        samples_file: string, optional
            The name of the file containing text samples.
            Default is `samples.txt`.
        dialect_labels_file: string, optional
            The name of the file containing dialect labels.
            Default is `dialect_labels.txt`.
        category_labels_file: string, optional
            The name of the file containing category labels.
            Default is `category_labels.txt`.
        """
        super(VarDialDataSet, self).__init__()
        self.data_dir = data_dir
        self.samples_file = samples_file
        self.dialect_labels_file = dialect_labels_file
        self.category_labels_file = category_labels_file

    @property
    def samples(self):
        """
        Returns a list of samples from the data set.
        """
        return [row['Sample'] for _, row in self.dataset.iterrows()]

    @property
    def ids(self):
        """
        Returns a list of sample ids.
        """
        return [row['Id'] for _, row in self.dataset.iterrows()]

    @property
    def category_labels(self):
        """
        Returns a list of category labels.
        """
        return [row['CategoryLabel'] for _, row in self.dataset.iterrows()]

    @property
    def dialect_labels(self):
        """
        Returns a list of dialect labels for each sample.
        """
        return [row['DialectLabel'] for _, row in self.dataset.iterrows()]

    def initialize(self):
        """
        Initializes the data set by loading all the data in memory.
        """
        logging.info('Loading category labels...')
        categ_df = pd.read_csv(path.join(self.data_dir,
                                         self.category_labels_file),
                               delimiter='\t',
                               header=None,
                               names=['Id', 'CategoryLabel'])

        logging.info('Loading dialect labels...')
        dialects_df = pd.read_csv(path.join(self.data_dir,
                                            self.dialect_labels_file),
                                  delimiter='\t',
                                  header=None,
                                  names=['Id', 'DialectLabel'])
        logging.info('Loading samples...')
        samples_df = pd.read_csv(path.join(self.data_dir, self.samples_file),
                                 delimiter='\t',
                                 header=None,
                                 names=['Id', 'Sample'])

        logging.info("Merging the data into the dataset...")
        ds = pd.merge(categ_df, dialects_df, on='Id')
        ds = pd.merge(ds, samples_df, on='Id')
        self.dataset = ds
        logging.info("Dataset initialized.")

    def preview(self, how='head', how_many=10):
        """
        Displays a preview of the data.

        Parameters
        ----------
        how: string, optional
            How to preview data. Options are 'head' or 'tail'.
            Default is 'head'.
        how_many: number, optional
            How many rows to include in the preview.
            The default is 10.
        """
        if how == 'head':
            print(self.dataset.head(how_many))
        else:
            print(self.dataset.tail(how_many))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    ds = VarDialDataSet('/data')
    ds.initialize()
    ds.preview()

    from itertools import groupby

    logging.info("Computing dataset stats...")
    print("Dataset statistics")
    print("Total samples: {}".format(len(ds.samples)))
    print("Samples per dialect label:")
    for key, collection in groupby(sorted(ds.dialect_labels)):
        print("Dialect {}: {}".format(key, len(list(collection))))
    print("Samples per category label:")
    for key, collection in groupby(sorted(ds.category_labels)):
        print("Category {}: {}".format(key, len(list(collection))))
    logging.info("That's all folks!")
