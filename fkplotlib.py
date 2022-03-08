import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from matplotlib.colors import ListedColormap
from cycler import cycler
from scipy.ndimage import gaussian_filter
import astropy.units as u
import os
import json


def colorline(
    x,
    y,
    z=None,
    ax=None,
    cmap=plt.get_cmap("copper"),
    norm=plt.Normalize(0.0, 1.0),
    linewidth=3,
    alpha=1.0,
):
    """
    Sources :
    https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    -----
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )

    if ax is None:
        ax = plt.gca()

    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Sources :
    https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
    -----
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def ax_legend(ax, *args, negative=False, **kwargs):
    """
    Add a simple, good-looking legend to your plots.
    """
    if not negative:
        leg = ax.legend(
            facecolor="w", frameon=True, edgecolor="k", framealpha=1, **kwargs
        )
    else:
        leg = ax.legend(
            facecolor="k", frameon=True, edgecolor="w", framealpha=1, **kwargs
        )
    leg.get_frame().set_linewidth(plt.rcParams["axes.linewidth"])


def axlegend(*args, **kwargs):
    ax_legend(*args, **kwargs)


def ax_bothticks(ax):
    """
    Add ticks on the top and right axis
    """
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")


def get_cmap(name):
    try:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), name + ".npy")
        cmap = ListedColormap(np.load(path))
    except:
        raise Exception("Colormap not recognized: " + path)
    return cmap


def use_txfonts():
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{txfonts}")


def get_errorbar_kwargs():
    return {
        "fmt": "o",
        "mec": "k",
        "ecolor": "k",
        "ms": 4,
        "capsize": 2,
        "elinewidth": 1.0,
        "capthick": 1.0,
        "mew": 1.0,
    }


def get_gruvbox_colors():
    with open(os.path.dirname(os.path.realpath(__file__)) + "/gruvbox.txt", "r") as f:
        colors = json.load(f)
    return colors


def get_gruvbox_cycler(which="bright"):
    colors = get_gruvbox_colors()
    which = which.lower()
    if which in ["bright", "neutral", "faded"]:
        col = colors[which]
        cyc = cycler(color=[col[k] for k in col.keys()])
    elif which == "all":
        ls = []
        for k in colors["bright"].keys():
            ls += [colors["bright"][k], colors["neutral"][k], colors["faded"][k]]
        cyc = cycler(color=ls)
    return cyc


def cmap_from_colors(col_list):
    import matplotlib as mpl
    from scipy.interpolate import interp1d

    x = np.arange(256)
    n = len(col_list)
    rgb_list = np.array([mpl.colors.to_rgb(col) for col in col_list])
    x_list = np.linspace(0.0, 255.0, n)
    rgb_interp = np.array(
        [interp1d(x_list, rgb_list[:, i], kind="linear")(x) for i in range(3)]
    )
    return mpl.colors.ListedColormap(rgb_interp.T)
