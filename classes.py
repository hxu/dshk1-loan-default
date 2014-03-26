import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


TRAIN_FILE = './data/train_v2.csv'
TEST_FILE = './data/test_v2.csv'


def get_train_data():
    df = pd.read_csv(TRAIN_FILE, na_values='NA', index_col='id')
    return df.iloc[:, 0:-1], df['loss']


class RemoveNoVarianceColumn(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.mask_ = [len(x.unique()) != 1 for n, x in X.iteritems()]
        return self

    def transform(self, X, y=None):
        return X.loc[:, self.mask_]


class RemoveObjectColumns(BaseEstimator, TransformerMixin):
    """
    Given a df, remove all columns of type object
    """
    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        self.mask_ = X.dtypes != np.object
        return self

    def transform(self, X):
        return X.loc[:, self.mask_]
