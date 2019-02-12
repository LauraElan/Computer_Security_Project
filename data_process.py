from sklearn import tree
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
import pickle as pkl
import pandas as pd
import glog as log
import numpy as np
import graphviz
import math


def pklDataPortion(X, Y, featureNames, pklname, portion=0.1):
    """Given entire dataset (X, Y), randomly pick @portion% of the data from it,
    and pickles the sampled data into the file named @pklname.
    Args:
        X, a numpy matrix of size #samples x #features.
        Y, an numpy array of size #samples.
        featureNames, the corresponding name of X's each column.
        pklname, where the pickled datas should be written to.
        portion, pickle only part of the dataset.
    """
    raise NotImplementedError()


def processDataset(filename, hasLabel=True, portions=[0.1]):
    """Properly process the .csv file named @filename.
    Args:
        filename: name of the .csv file, NOT including the '.csv' extension.
        hashLabel: true if the file has label (training data), otherwise false.
        portions: list of floats, at the end of the processing,
                  for each x in portions, pickling 100*x% of the data.
    Returns:
        X, a numpy matrix of size #samples x #features
        Y, if @hasLabel is True, an numpy array of size #samples, otherwise None.
        featureNames, the corresponding name of X's each column.
    """
    raise NotImplementedError()


def loadPkl(filename):
    """Load dataset (X, Y, featureNames) from pre-cached pickle file.
    Returns:
        X, feature matrix.
        Y, label vector.
        featureNames, column names for X.
    """
    raise NotImplementedError()


if __name__ == '__main__':
    """How your processDataset will be called from main function"""
    processDataset('train_samples', hasLabel=True, portions=[1.0])
    processDataset('train', hasLabel=True, portions=[0.01, 0.1])
