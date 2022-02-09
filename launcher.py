import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import json
import fkplotlib
import cluster_catalog
from fisc import Data
from joblib import Parallel, delayed
from copy import copy

fkplotlib.use_txfonts()
# plt.ion()

"""
In this code, I follow the LIRA convention, where
the equation of a line is  y = alpha + beta * x
"""

# boxes = [
#   cluster_catalog.Box(0.5, 0.9, 0.21, 2.2),  # 1 big LPSZ box
# ]
boxes = cluster_catalog.boxes_lpsz
subsample_size = 50
n_fits = 10000
n_threads = 50

truth = {"alpha": -0.19, "beta": 1.79, "sigma": 0.075}  # Planck
path_to_results = f"./Results_24/"

def process(i, do_plots=False):

    print(f"======> Launching {i}...")
    # ======== Selection of a subsample
    new_catalog = catalog.select_from_boxes(
        boxes, subsample_size // len(boxes) * np.ones(len(boxes), dtype=int)
    )

    # ======== Fit scaling relation
    d = Data.from_table(
        new_catalog(),
        x_obs="log_M_tilde",
        y_obs="log_Y_tilde",
        x_err="err_log_M_tilde",
        y_err="err_log_Y_tilde",
        y_threshold="log_Y_tilde_threshold",
        corr="corr",
    )

    #d.y_threshold = np.ones(d.n_pts) * np.min(d.y_threshold)
    d.y_threshold = None
    d.set_axesnames(
        r"${\rm log}_{10} \, \frac{M_{500}}{\rm 6 \times 10^14 \; M_\odot}$",
        r"${\rm log}_{10} \, \frac{E^{-2/3}(z) \, {\cal D}_{\rm A}^2 Y_{500}}"
        + r"{\rm 10^{-4} \; Mpc^2}$",
    )

    for nmix in [1]:
        d.set_path_to_results(f"{path_to_results}/{i}/")
        d.to_table().to_pandas().to_csv(f"{d.path_to_results}/data.csv", index=False)
        try:
            d.fit_lira(
                nmix,
                5e4,
                lira_args={
                    #"alpha.YIZ": truth["alpha"],
                    "sigma.YIZ.0": "dunif(0.001, 1.0)",
                    #"sigma.YIZ.0": float(sigma),
                    "sigma.XIZ.0": 0.0,
                    "alpha.XIZ": "dunif(-0.5, 0.5)",
                },
            )
            d.lira_chains.to_csv(f"{d.path_to_results}/chains.csv", index=False)

        except Exception as e:
            print(e)

        if do_plots:
            d.plot_lira_results(
                nmix=nmix, chains_file=f"{d.path_to_results}/chains.csv"
            )
        print(f"Results stored in {d.path_to_results}")
    return d


# ================================================================================ #
# ================================================================================ #

t = Table.read("catalog_universe_1e5.fits", 1)

if n_threads != 1:
    catalog = cluster_catalog.ClusterCatalog.from_table(t)
    catalog.to_observable(**truth)
    catalog.measurement_errors(
        err_frac_Y=(0.1, 0.0),
        err_frac_M=(0.1, 0.0),
        corr=(0.8, 0.9),
        shake=True,
    )

    _ = Parallel(n_jobs=n_threads)(
        delayed(process)(i, do_plots=False) for i in range(n_fits)
    )
else:
    catalog = cluster_catalog.ClusterCatalog.from_table(t)
    catalog.to_observable(**truth)
    catalog.measurement_errors(
        err_frac_Y=(0.1, 0.0),
        err_frac_M=(0.1, 0.0),
        corr=(0.8, 0.9),
        shake=True,
    )

    d = [process(i) for i in range(n_fits)]
