import numpy as np
import pandas as pd
import pytest
from ..ml.model import train_model, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier
np.random.seed(0)

@pytest.fixture(scope="function")
def X_train():
    cols = 10
    rows = 100
    df = pd.DataFrame(
        data=np.random.randn(rows, cols),
        columns=[f'col_{i}' for i in range(cols)]
    )

    return df

@pytest.fixture(scope="function")
def y_train():
    rows = 100
    y = pd.Series(
        np.random.choice(2, rows),
        name='y'
    )

    return y

@pytest.fixture(scope="function")
def y_pred():
    rows = 100
    y = pd.Series(
        np.ones(rows),
        name='y_preds'
    )

    return y


def test_train_model(X_train, y_train):
    assert isinstance(train_model(X_train, y_train), RandomForestClassifier)


def test_compute_model_metrics(y_train, y_pred):
    p, r, f = compute_model_metrics(y_train, y_pred)
    assert isinstance(p, float) and 1. >= p >= 0.
    assert isinstance(r, float) and 1. >= r >= 0.
    assert isinstance(f, float) and f >= 0.
