import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from chainconsumer import ChainConsumer
import sys
import os

paths = sys.argv[1:]

meds_lira = [{"alpha.YIZ": [], "beta.YIZ": [], "sigma.YIZ.0": []} for _ in paths]
stds_lira = [{"alpha.YIZ": [], "beta.YIZ": [], "sigma.YIZ.0": []} for _ in paths]
params_toplot = list(meds_lira[0].keys())


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


for i, path in enumerate(paths):
    if path[-1] != "/":
        path += "/"
    dirs = sorted([f for f in os.listdir(path) if os.path.isdir(path + f)])
    for ii in dirs:
        _dir = f"{path}/{ii}/"
        print(_dir)
        try:
            chains_lira = pd.read_csv(f"{_dir}/chains.csv", sep=",")
        except:
            print(f"No data for {ii}")
            continue

        for p in params_toplot:
            meds_lira[i][p].append(np.median(chains_lira[p]))
            stds_lira[i][p].append(np.std(chains_lira[p], ddof=1))

    pd.DataFrame(meds_lira[i]).to_csv(path + "/medians.csv", index=False)
    pd.DataFrame(stds_lira[i]).to_csv(path + "/stdevs.csv", index=False)
