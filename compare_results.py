import numpy as np
import matplotlib.pyplot as plt
import fkplotlib
import pandas as pd
import json
from copy import copy
from chainconsumer import ChainConsumer
import sys
import os

fkplotlib.use_txfonts()
mycmap = "Spectral" #fkplotlib.get_cmap("better_spectral")

# arg = sys.argv[1]
# names = sorted(os.listdir(arg))
# paths = [os.path.join(arg, name) for name in names]

names = sorted(sys.argv[1:-1])
paths = names
which_estimator = sys.argv[-1]

# results_lira = [{"alpha.YIZ": [], "beta.YIZ": [], "sigma.YIZ.0": []} for _ in paths]
results_lira = [{"alpha.YIZ": [], "beta.YIZ": []} for _ in paths]
truth = [-0.19, 1.79]


def rename_params(cc):
    for chain in cc.chains:
        for i, p in enumerate(chain.parameters):
            new_p = r"$" + p + r"$"
            new_p = new_p.replace("alpha", r"\alpha")
            new_p = new_p.replace("beta", r"\beta")
            new_p = new_p.replace("sigma", r"\sigma")
            new_p = new_p.replace(".YIZ", r"_{Y|Z}")
            new_p = new_p.replace(".XIZ", r"_{X|Z}")
            new_p = new_p.replace(".0", "")
            chain.parameters[i] = new_p


for lira, path in zip(results_lira, paths):
    if path[-1] != "/":
        path += "/"
    if os.path.isfile(f"{path}/{which_estimator}.csv"):
        chains = pd.read_csv(f"{path}/{which_estimator}.csv")
        msk = chains["alpha.YIZ"] > -10.0
        if np.std(chains["sigma.YIZ.0"]) > 1e-6:
            lira["sigma.YIZ.0"] = []
        for p in lira.keys():
            lira[p] = np.array(chains[p][msk])
            # lira[p] = np.array(chains[p])
    # else:
    #    dirs = sorted([f for f in os.listdir(path) if os.path.isdir(path + f)])
    #    for i in dirs:
    #        _dir = f"{path}/{i}/"
    #        print(_dir)
    #        try:
    #            chains_lira = pd.read_csv(f"{_dir}/chains.csv", sep=",")
    #        except:
    #            print(f"No data for {i}")
    #            continue

    #        for p in params_toplot:
    #            lira[p].append(np.nanmedian(chains_lira[p]))

    #    df = pd.DataFrame(lira)
    #    df.to_csv(path + "medians.csv", index=False)

plt.style.use("dark_background")

cc = ChainConsumer()
for lira, name in zip(results_lira, names):
    cc.add_chain(lira, name=name.replace("_", r"\_"), shade_alpha=0, shade_gradient=0)
cc.configure(cmap=mycmap, sigmas=[1])
cc.configure_truth(color="#FFFFFF", linestyle=":", alpha=0.5, linewidth=2)
rename_params(cc)
cfig = cc.plotter.plot(
    figsize=(7, 7),
    truth=truth,
    extents=[(-0.19 - 0.25, -0.19 + 0.25), (1.79 - 3.0, 1.79 + 1.0)],
)
cfig.subplots_adjust(hspace=0, wspace=0)
cfig.align_labels()
cfig.savefig("corner.pdf")

cfig = cc.plotter.plot_summary(
    truth=truth,
    #extents=[(-0.19 - 0.25, -0.19 + 0.25), (1.79 - 3.0, 1.79 + 1.0)],
)
cfig.align_labels()
cfig.savefig("summary.pdf")
