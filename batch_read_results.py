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

path = sys.argv[1]
if path[-1] != "/":
    path += "/"
do_indiv = "--indiv" in sys.argv

cc_big = ChainConsumer()
dirs = [f for f in os.listdir(path) if os.path.isdir(path + f)]

results_lira = {"alpha.YIZ": [], "beta.YIZ": [], "sigma.YIZ.0": [], "sigma.XIZ.0": []}


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


for i in dirs:
    _dir = f"{path}/{i}/"
    print(_dir)
    try:
        chains_lira = pd.read_csv(f"{_dir}/chains.csv", sep=",")
    except:
        print(f"No data for {i}")
        continue

    params_toplot = ["alpha.YIZ", "beta.YIZ", "sigma.YIZ.0"]
    if not np.all(chains_lira["sigma.XIZ.0"] == chains_lira["sigma.XIZ.0"][0]):
        params_toplot.append("sigma.XIZ.0")

    # Individual corner plot
    if do_indiv:
        cc = ChainConsumer()
        cc.add_chain(
            chains_lira[params_toplot],
            shade=True,
            shade_alpha=0.3,
            shade_gradient=0.0,
            name="LIRA",
        )
        cc.configure(cmap="RdBu_r", sigmas=[1], summary=False)
        rename_params(cc)
        cfig = cc.plotter.plot()
        cfig.subplots_adjust(
            left=0.15, right=0.85, top=0.85, bottom=0.15, wspace=0, hspace=0
        )
        cfig.align_labels()
        cfig.savefig(f"{_dir}/corner.pdf")
        plt.close(cfig)
        wfig = cc.plotter.plot_walks()
        wfig.subplots_adjust(
            left=0.15, right=0.85, top=0.85, bottom=0.15, wspace=0, hspace=0
        )
        wfig.align_labels()
        wfig.savefig(f"{_dir}/walks.pdf")
        plt.close(wfig)

    # All corner plot
    cc_big.add_chain(
        chains_lira[params_toplot],
        name=i,
        plot_point=False,
        shade=True,
        shade_alpha=0.0,
        linewidth=1.0,
    )

    for p in params_toplot:
        results_lira[p].append(np.nanmedian(chains_lira[p]))

plt.style.use("dark_background")
cc_big.configure(
    cmap="Spectral",
    sigmas=[1],
    shade=True,
    shade_alpha=1.0,
    plot_point=False,
    plot_contour=True,
)
rename_params(cc_big)
cc_big.configure_truth(linestyle=":", linewidth=2, color="#FFFFFF", alpha=0.3)
cfig_big = cc_big.plotter.plot_distributions(
    figsize=(7, 4),
    truth=[-0.19, 1.79, 0.025],
)
cfig_big.align_labels()
cfig_big.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2, wspace=0)
cfig_big.savefig(f"{path}/distributions.pdf")

params_toplot = copy(list(results_lira.keys()))
for p in params_toplot:
    if len(results_lira[p]) == 0:
        _ = results_lira.pop(p)
cc = ChainConsumer()
cc.add_chain(
    results_lira,
    name="LIRA",
    shade_alpha=0.0,
    shade_gradient=0,
    color="#FFFFFF",
    bar_shade=False,
)
cc.configure(cmap="RdBu_r", sigmas=[1], summary=False)
rename_params(cc)
cc.configure_truth(linestyle=":", linewidth=2, color="#FFFFFF", alpha=0.3)
cfig = cc.plotter.plot(
    figsize=(7, 7),
    truth=[-0.19, 1.79, 0.025],
)
cfig.align_labels()
cfig.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15, wspace=0, hspace=0)
cfig.savefig(f"{path}/corner.pdf")
