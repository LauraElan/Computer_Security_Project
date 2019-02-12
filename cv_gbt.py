import pickle as pkl
import pandas as pd
import glog as log
import numpy as np
import xgboost
from data_process import loadPkl
from cross_valid import crossValidate, rankFeatures


def trainGradientBoost(clf, trainX, trainY, validX, validY,
                       featureNames, k=0, top=5):
    """How to train a gradient boosting classifier.
    Args:
        clf: expect to be a sklearn GradientBoostingClassifier object.
        (trainX, trainY, validX, validY): splitted train/valid dataset.
        featureNames: column names of @trainX and @validX.
        k: which round of cross validation.
        top: print top-x features.
    Returns:
        clf: fitted classifier.
        trainAccu: final training accuracy.
        validAccu: final validation accuracy.
    """
    raise NotImplementedError()


if __name__ == '__main__':
    """Write code to perform CV on a GradientBoostingClassifier."""
