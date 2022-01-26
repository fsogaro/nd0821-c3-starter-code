import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from collections import defaultdict

from .data import process_data, slice_data_on_category

logger = logging.getLogger()

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # todo: get  different models
    mdl = RandomForestClassifier()
    # optimise best parameters
    # fit best parameters
    scores = cross_val_score(mdl, X_train, y_train, cv=5)
    logging.info(f"*** cv score mean:{np.mean(scores)}  ***")
    logging.info(f"*** cv score std:{np.std(scores)}  ***")

    mdl.fit(X_train, y_train)
    # return model
    return mdl


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def categorical_slice_performance(data, mdl, encoder, lb, cat_features):

    performance_sliced = defaultdict(dict)
    for feature in cat_features:
        feature_categ = data[feature].unique()
        perf_categ = defaultdict(float)

        for category in feature_categ:
            sliced = slice_data_on_category(data, feature, category)
            X_slice, y_slice, _, _ = process_data(
                sliced,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb
            )
            preds = inference(mdl, X_slice)
            perf_categ[category], r, f = compute_model_metrics(y_slice, preds)
        performance_sliced[feature] = perf_categ

    return performance_sliced




