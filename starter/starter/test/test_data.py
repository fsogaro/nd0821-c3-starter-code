import numpy as np
import pandas as pd
import pytest
from ..ml.data import slice_data_on_category
np.random.seed(0)

@pytest.fixture(scope="function")
def data():

    df = pd.DataFrame(
        {
            "sex": ["M", "F", "M"],
            "age": [20, 30, 40],
            "edu": ["uni", "prim", "mid"],
        }
    )

    return df


def test_slice_data_on_category(data):

    out = slice_data_on_category(data, column="sex", category="M")
    assert out.shape == (2, 3)

    out = slice_data_on_category(data, column="edu", category="uni")
    assert out.shape == (1, 3)
