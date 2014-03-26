from sklearn.base import BaseEstimator, TransformerMixin
import classes
import numpy as np
import pandas as pd


X, y = classes.get_train_data()

desc = X.describe()
desc.iloc[:, 0:5]


n_unique = [len(x.unique()) for n, x in X.iteritems()]

n_na = [sum(x == np.nan) for n, x in X.iteritems()]
n_na = [sum(pd.isnull(x)) for n, x in X.iteritems()]



X = classes.get_train_data()
remove_cols = RemoveNoVarianceColumn()
remove_cols.fit(X)
X = remove_cols.transform(X)

test_x = classes.get_test_data()
transformed_test_x = remove_cols.transform(test_x)
