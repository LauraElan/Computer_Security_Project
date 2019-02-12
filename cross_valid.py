from sklearn.model_selection import StratifiedKFold
import pickle as pkl
import pandas as pd
import glog as log
import numpy as np


def rankFeatures(clf, featureNames, top=10):
    """Given a trained classifier that is able to rank features,
    sort the feature names in @featureNames, and print several top features.
    Args:
        cls: a fitted classifier having the scores of feature importances.
        featureNames: names of the features.
        top: how many top features to print to the stdout.
    Returns:
        sortedFeatures, feature names sorted by its importance.
        sortedFeatureImportances, the scores for each item in sortedFeatures.
    """
    raise NotImplementedError()


def crossValidate(clf, X, Y, featureNames, trainFunc, fold=5):
    """Cross validate using stratified k-fold split.
    Args:
        clf: a sklearn classifier of any type.
        (X, Y): feature matrix and label vector from the dataset.
        featureNames: column names of @X.
        tranFunc: a function object that implements how to fit the @clf.
        fold: number of fold to run for cross validation.
    Returns:
        Metrics on each validation fold as a dictionary, e.g.
            result['ValidAccuList'] = [accu_1, accu_2, ..., accu_k]
            result['ValidLossList'] = [loss_1, loss_2, ..., loss_k]
            ...
    """
    raise NotImplementedError()
