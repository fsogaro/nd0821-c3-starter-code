
import pickle
from typing import Any


def save_pickle(model: Any, filename: str) -> None:
    """

    Parameters
    ----------
    model
    filename

    Returns
    -------

    """
    with open(filename, 'wb') as files:
        pickle.dump(model, files)
