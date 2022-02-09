import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from fisc import Data


def mksample(n, truth, threshold=None):
    Z = np.random.normal(0.0, 1.0, n)
    X = np.random.normal(
        truth["alpha.XIZ"] + truth["beta.XIZ"] * Z, truth["sigma.XIZ.0"]
    )
    Y = np.random.normal(
        truth["alpha.YIZ"] + truth["beta.YIZ"] * Z, truth["sigma.YIZ.0"]
    )

    err_x = np.ones(n) * 0.1
    err_y = np.ones(n) * 0.1
    x = np.random.normal(X, err_x)
    y = np.random.normal(Y, err_x)
    msk = (y > threshold) if (threshold is not None) else (np.ones(n, dtype=bool))
    print("Data size:", np.sum(msk))
    return x[msk], y[msk], err_x[msk], err_y[msk]


def do_fit(i, n, truth, lira_args, threshold=None):
    x, y, err_x, err_y = mksample(n, truth, threshold=threshold)
    d = Data(x, y, err_x, err_y, y_threshold=threshold)
    # fig, ax = d.plot_data()
    chains = d.fit_lira(1, 5000, lira_args=lira_args)
    meds, stds = chains.median(), chains.std()
    print(f"========== Finished {i} ==========")
    return (
        {p: meds[p] for p in ["alpha.YIZ", "beta.YIZ", "sigma.YIZ.0"]},
        {p: stds[p] for p in ["alpha.YIZ", "beta.YIZ", "sigma.YIZ.0"]},
    )


# ^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^ #
# v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v #

'''
# ===== 1) Simplest case ===== #
np.random.seed(42)
n = 100
truth = {
    "alpha.YIZ": 1.0, "beta.YIZ": 1.0, "sigma.YIZ.0": 0.1,
    "alpha.XIZ": 0.0, "beta.XIZ": 1.0, "sigma.XIZ.0": 0.0,
}

results = Parallel(n_jobs=50)(
    delayed(do_fit)(
        i, n, truth, lira_args={"sigma.YIZ.0": "dunif(1e-4, 1e4)", "sigma.XIZ.0": 0.0}
    )
    for i in range(5000)
)
meds = pd.DataFrame([res[0] for res in results])
stds = pd.DataFrame([res[1] for res in results])
meds.to_csv("Results_simple_0/medians.csv", index=False)
stds.to_csv("Results_simple_0/stdevs.csv", index=False)

# ===== 2) Scattered mass proxy ===== #
np.random.seed(42)
n = 100
truth = {
    "alpha.YIZ": 1.0,
    "beta.YIZ": 1.0,
    "sigma.YIZ.0": 0.1,
    "alpha.XIZ": 0.0,
    "beta.XIZ": 1.0,
    "sigma.XIZ.0": 0.1,
}

results = Parallel(n_jobs=50)(
    delayed(do_fit)(
        i,
        n,
        truth,
        lira_args={
            "sigma.YIZ.0": "dunif(1e-4, 1e4)",
            "sigma.XIZ.0": "dnorm(0.1, 0.01)",
        },
    )
    for i in range(5000)
)
meds = pd.DataFrame([res[0] for res in results])
stds = pd.DataFrame([res[1] for res in results])
meds.to_csv("Results_simple_1/medians.csv", index=False)
stds.to_csv("Results_simple_1/stdevs.csv", index=False)

# ===== 3) Biased mass proxy ===== #
np.random.seed(42)
n = 100
truth = {
    "alpha.YIZ": 1.0, "beta.YIZ": 1.0, "sigma.YIZ.0": 0.1,
    "alpha.XIZ": 0.8, "beta.XIZ": 1.0, "sigma.XIZ.0": 0.0,
}

results = Parallel(n_jobs=50)(
    delayed(do_fit)(
        i,
        n,
        truth,
        lira_args={
            "sigma.YIZ.0": "dunif(1e-4, 1e4)",
            "sigma.XIZ.0": 0.0,
            "alpha.XIZ": "dunif(0.7, 0.9)",
        },
    )
    for i in range(5000)
)
meds = pd.DataFrame([res[0] for res in results])
stds = pd.DataFrame([res[1] for res in results])
meds.to_csv("Results_simple_2/medians.csv", index=False)
stds.to_csv("Results_simple_2/stdevs.csv", index=False)
'''

# ===== 4) Cut in y ===== #
np.random.seed(42)
n = 200
truth = {
    "alpha.YIZ": 1.0, "beta.YIZ": 1.0, "sigma.YIZ.0": 0.1,
    "alpha.XIZ": 0.0, "beta.XIZ": 1.0, "sigma.XIZ.0": 0.0,
}

results = Parallel(n_jobs=50)(
    delayed(do_fit)(
        i,
        n,
        truth,
        lira_args={"sigma.YIZ.0": "prec.dgamma"},
        threshold=1.0
    )
    for i in range(5000)
)
meds = pd.DataFrame([res[0] for res in results])
stds = pd.DataFrame([res[1] for res in results])
meds.to_csv("Results_simple_3/medians.csv", index=False)
stds.to_csv("Results_simple_3/stdevs.csv", index=False)
