from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn_pandas import cross_val_score
import classes
import numpy as np
import pandas as pd


def logistic_001():
    X, y = classes.get_train_data()
    y = y > 0

    remove_object = classes.RemoveObjectColumns()
    X = remove_object.fit_transform(X)

    imputer = Imputer()
    X = imputer.fit_transform(X)
    scores = []

    for i in range(X.shape[1]):
        clf = LogisticRegression()
        s = cross_val_score(clf, X[:, i], y, scoring='roc')
        scores.append((i, s))
