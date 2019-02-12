from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
import pandas as pd
import glog as log
import numpy as np
import graphviz
import xgboost
from data_process import loadPkl
from cross_valid import crossValidate, rankFeatures


def trainRandomForest(clf, trainX, trainY, validX, validY,
                      featureNames, k=0, top=10):
    """How to train a random forest classifier.
    Args:
        clf: expect to be a sklearn RandomForestClassifier object.
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
    """Write code to perform CV on a RandomForestClassifier."""
