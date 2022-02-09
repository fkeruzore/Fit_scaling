import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from iminuit import Minuit
import fkplotlib
import pandas as pd
import json
from copy import copy
from chainconsumer import ChainConsumer
import sys
import os

fkplotlib.use_txfonts()

paths = sys.argv[1:]

results_lira = [{"alpha.YIZ": [], "beta.YIZ": []} for _ in paths]
results_lira = [{"alpha.YIZ": [], "beta.YIZ": [], "sigma.YIZ.0": []} for _ in paths]
params_toplot = list(results_lira[0].keys())


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
    dirs = sorted([f for f in os.listdir(path) if os.path.isdir(path + f)])
    for i in dirs:
        _dir = f"{path}/{i}/"
        print(_dir)
        try:
            chains_lira = pd.read_csv(f"{_dir}/chains.csv", sep=",")[params_toplot]
            kde = gaussian_kde(np.array(chains_lira).T)
        except:
            print(f"No data for {i}")
            continue

        def pseudochi2(args):
            return -2.0 * kde.logpdf(args)[0]

        m = Minuit.from_array_func(
            pseudochi2, [-0.19, 1.79, 0.075], error=[0.1, 0.1, 0.1], errordef=1
        )
        res = m.migrad()

        for i, p in enumerate(params_toplot):
            lira[p].append(res.params[i].value)

    df = pd.DataFrame(lira)
    df.to_csv(path + "/bestfits.csv", index=False)
