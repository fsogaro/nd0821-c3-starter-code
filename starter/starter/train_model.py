# Script to train machine learning model.
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics,\
    categorical_slice_performance
from ml.utils import save_pickle, save_df_as_image
# Add the necessary imports for the starter code.
# Add code to load in the data.
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def develop_model():
    data = pd.read_csv("../data/census_int.csv")
    save_path = os.path.join("..", "model")
    logging.info(f"*** loaded data: shape {data.shape} ***")
    logging.info(f"*** loaded data: columns {data.columns} ***")

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
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    logging.info(f"*** processing test ***")
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    logging.info(f"*** saving encoders  ***")

    save_pickle(encoder, os.path.join(save_path, "hot_encoder.pickle"))
    save_pickle(lb, os.path.join(save_path, "label_encoder.pickle"))

    # Train and save a model.
    logging.info(f"*** training model ***")
    mdl = train_model(X_train, y_train)
    logging.info(f"*** save models  ***")
    save_pickle(mdl, os.path.join(save_path, "trained_model.pickle"))

    y_test_pred = mdl.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_test_pred)
    logging.info(f"*** METRICS: ***")
    logging.info(f"\n precision: {precision},\n recall: {recall},\n fbeta:"
                 f" {fbeta}")

    logging.info("*** computing performance on slices of test data ***")
    slice_perf_dict = categorical_slice_performance(
        test, mdl, encoder, lb, cat_features)

    save_df_as_image(pd.DataFrame(slice_perf_dict),
                     os.path.join("..", "screenshots", "sliced_performances"))


if __name__ == "__main__":
    develop_model()
