# Script to train machine learning model.
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
from ml.utils import save_pickle
# Add the necessary imports for the starter code.
# Add code to load in the data.

def go():
    data = pd.read_csv("data/census.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    logging.info(f"*** processing train ***")

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    logging.info(f"*** processing test ***")
    X_test, y_test, _, _ = process_data(
        train,
        categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    logging.info(f"*** saving encoders  ***")

    save_pickle(encoder, "hot_encoder.pickle")
    save_pickle(lb, "label_encoder.pickle")

    # Train and save a model.
    logging.info(f"*** training model ***")
    mdl = train_model(X_train, y_train)
    logging.info(f"*** save models  ***")
    save_pickle(mdl, "trained_model.pickle")

    y_test_pred = mdl.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_test_pred)
    logging.info(f"*** METRICS: ***")
    logging.info(f"precision: {precision},\n recall: {recall},\n fbeta: {fbeta}")


if __name__ == "__main__":
    go()