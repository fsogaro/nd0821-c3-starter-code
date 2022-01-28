
import pickle
from typing import Any
import seaborn as sns
from matplotlib import pyplot as plt


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

def load_pickle(file):
    """

        Parameters
        ----------
        filename

        Returns
        -------

        """
    with open(file, "rb") as f:
        pickle_object = pickle.load(f)
    return pickle_object

def save_df_as_image(df, path):
    # Set background to white
    norm = matplotlib.colors.Normalize(-1,1)
    colors = [[norm(-1.0), "white"],
            [norm( 1.0), "white"]]
    cmap = plt.colors.LinearSegmentedColormap.from_list("", colors)
    # Make plot
    plot = sns.heatmap(df, annot=True, cmap=cmap, cbar=False)
    fig = plot.get_figure()
    fig.set_size_inches(15, 20)
    fig.savefig(path)