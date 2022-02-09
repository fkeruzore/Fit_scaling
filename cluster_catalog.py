import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RectBivariateSpline
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from copy import copy
import fkplotlib


class Box:
    """
    A "box", i.e. a rectangle of redshift and Y_tilde boundaries.

    Args:
        z_inf (float): lower limit on redshift,
        z_sup (float): higher limit on redshift,
        y_inf (float): lower limit on Y_tilde,
        y_sup (float): higher limit on Y_tilde,
    """

    def __init__(self, z_inf, z_sup, y_inf, y_sup):
        self.z_inf = z_inf
        self.z_sup = z_sup
        self.y_inf = y_inf
        self.y_sup = y_sup

    def __str__(self):
        return f"{self.z_inf} < z < {self.z_sup} ; {self.y_inf} < y < {self.y_sup}"

    def is_in_box(self, y, z):
        """
        Evaluates whether or not a set of (y, z) are in the box.

        Args:
            y (float or array-like): value(s) of y to test;
            z (float or array-like): value(s) of z to test;

        Returns:
            (bool or srray-like) whether the value(s) tested
                are in the box or not
        """

        return np.logical_and(
            np.logical_and(y >= self.y_inf, y <= self.y_sup),
            np.logical_and(z >= self.z_inf, z <= self.z_sup),
        )

    def plot_on_ax(self, ax, **kwargs):
        """
        Plots the limits of the box on a (z, y) plot.

        Args:
            ax (``matplotlib.pyplot.Axes``): the axis in which to
                overlay the box plot;

        Notes:
            ``**kwargs`` are passed to ``lines``
        """
        ax.vlines([self.z_inf, self.z_sup], self.y_inf, self.y_sup, **kwargs)
        ax.hlines([self.y_inf, self.y_sup], self.z_inf, self.z_sup, **kwargs)


boxes_lpsz = [
    Box(
        [0.5, 0.7, 0.9][i],
        [0.5, 0.7, 0.9][i + 1],
        [0.21, 0.335, 0.55, 0.88, 1.4, 2.2][j],
        [0.21, 0.335, 0.55, 0.88, 1.4, 2.2][j + 1],
    )
    for i in range(2)
    for j in range(5)
]


class ClusterCatalog:
    def __init__(self, z, M_500, boxes=[], **kwargs):
        self.z = z
        self.M_500 = M_500
        self.qts = ["z", "M_500"]  # available quantities

        for kw in kwargs:
            self.__dict__[kw] = kwargs[kw]
            self.qts.append(kw)

        self.boxes = boxes

    # ============================================================================ #

    def to_table(self, columns=None):
        if columns is None:
            columns = self.qts
        table = Table([self.__dict__[attr] for attr in columns], names=columns)
        return table

    def __str__(self):
        return self.to_table().__str__()

    def __call__(self):
        return self.to_table()

    def __getitem__(self, key):
        return self.to_table()[key]

    # ============================================================================ #

    @classmethod
    def from_table(cls, table, boxes=[]):
        colnames_toadd = copy(table.colnames)
        try:
            colnames_toadd.remove("z")
            colnames_toadd.remove("M_500")
        except Exception as e:
            raise Exception(f"`z` and `M_500` must be in the table ({e})")
        inst = cls(
            table["z"].data,
            table["M_500"].data * table["M_500"].unit,
            boxes=boxes,
            **{c: table[c] for c in colnames_toadd},
        )
        inst.n = len(table)
        return inst

    # ============================================================================ #

    @classmethod
    def from_distribution(
        cls,
        n,
        distrib_M=np.random.uniform,
        distrib_z=np.random.uniform,
        limits_M=(1e14, 2e15),
        limits_z=(1e-3, 1.5),
    ):
        """
        Creates a catalog randomly drawn from a distribution.

        Args:
            n (int): number of clusters to draw;
            distrib_M (func): the distribution to be used for mass,
                see Notes;
            distrib_z (func): the distribution to be used for redshift,
                see Notes;
            limits_M (tuple): the mass limits to use in solar masses;
            limits_z (tuple): the redshift limits to use.

        Notes:
            ``distrib`` will be called via ``distrib(*limits_M, n)``
            in mass, and via ``distrib(*limits_z, n)`` in redshift.
            For example, for a uniform distribution (used when
            ``distrib=np.random.uniform``), ``limits_M`` will be the
            lower and upper limits in mass; for a normal distribution
            (used when ``distrib=np.random.normal``), it will be the
            mean and standard deviation.
        """
        t = Table(
            {
                "z": distrib_z(*limits_z, n),
                "M_500": distrib_M(*limits_M, n) * u.Unit("Msun"),
            }
        )
        inst = cls.from_table(t)

        return inst

    # ============================================================================ #

    @classmethod
    def from_massfunc(
        cls, n, limits_M=(1e14, 2e15), limits_z=(1e-3, 1.5), mass_func_file=None
    ):
        """
        Creates a sample of clusters that follows a mass function.

        Args:

            n (int): number of clusters,

            limits_M (tuple): the inf and sup limits in mass, in solar masses,

            limits_z (typle): the inf and sup limits in redshift,

            mass_func_file (str): if the mass function is already computed,
                a path to the .npz file where it has been saved,
        """
        n = int(n)

        if mass_func_file is not None:
            try:
                f = np.load(mass_func_file)
                d2n_dmdz = f["d2n_dmdz"]
                redshifts = f["z"]
                masses = f["m"]
            except Exception:
                pass
        else:
            import hmf

            mf = hmf.MassFunction(
                Mmin=np.log10(limits_M[0]),
                Mmax=np.log10(limits_M[1]),
                mdef_model="SOCritical",
                mdef_params={"overdensity": 500.0},
            )
            mass_func_file = "mass_func.npz"

            def nb_of_haloes(z):
                mf.z = z
                return mf.dndm * cosmo.comoving_volume(z).to("Mpc3").value

            redshifts = np.linspace(*limits_z, 500)
            masses = mf.m

            d2n_dmdz = np.array([nb_of_haloes(z) for z in redshifts]).T
            np.savez(mass_func_file, d2n_dmdz=d2n_dmdz, m=masses, z=redshifts)

        mass_func_interpolator = RectBivariateSpline(
            redshifts, masses, np.log(d2n_dmdz.T), kx=1, ky=1
        )

        def interp_massfunc(z, M):
            return np.exp(mass_func_interpolator(z, M, grid=False))

        catalog = Table(names=["z", "M_500"])
        n_draws = 10000
        while len(catalog) < n:
            rd_pts = {
                "z": np.random.uniform(redshifts.min(), redshifts.max(), n_draws),
                "M": np.random.uniform(masses.min(), masses.max(), n_draws),
                "d2n_dMdz": np.random.uniform(d2n_dmdz.min(), d2n_dmdz.max(), n_draws),
            }
            are_sel = interp_massfunc(rd_pts["z"], rd_pts["M"]) > rd_pts["d2n_dMdz"]
            for i in np.where(are_sel)[0]:
                catalog.add_row([rd_pts["z"][i], rd_pts["M"][i]])
        catalog = catalog[0:n]
        catalog["M_500"].unit = u.Msun

        inst = cls.from_table(catalog)
        inst.load_mass_func_file(mass_func_file)

        return inst

    # ============================================================================ #

    def to_observable(self, alpha=-0.19, beta=1.79, sigma=0.075):
        """
        Y = alpha + beta X +- sigma, with
        Y = Ez^(-2/3) Da2 Y500 / 1e-4 Mpc2
        X = M500 / 6e14 Msun
        """

        M_tilde = (self.M_500.to("Msun") / (6e14 * u.Msun)).value
        log_M_tilde = np.log10(M_tilde)
        log_Y_tilde = np.random.normal(alpha + beta * log_M_tilde, sigma)
        Y_tilde = 10 ** log_Y_tilde
        Y_500 = Y_tilde * cosmo.efunc(self.z) ** (2.0 / 3.0) * 1e-4 * u.Unit("Mpc2")

        self.M_tilde = M_tilde
        self.log_M_tilde = np.log10(M_tilde)
        self.Y_tilde = Y_tilde
        self.log_Y_tilde = np.log10(Y_tilde)
        self.Y_500 = Y_500

        for qty in ["M_tilde", "log_M_tilde", "Y_tilde", "log_Y_tilde", "Y_500"]:
            if qty not in self.qts:
                self.qts.append(qty)

    # ============================================================================ #

    def measurement_errors(
        self,
        err_frac_Y=None,
        err_frac_M=None,
        corr=None,
        shake=True,
        sigma_scatter_M=0.0,
    ):
        """
        Add measurement errors.

        Args:
            err_frac_Y (tuple): mean and std of the error bars on integrated
                SZ signal, in fraction of the central value;
            err_frac_M (tuple): mean and std of the error bars on mass,
                in fraction of the central value;
            corr (tuple): inf and sup limits on the correlation between
                errors on Y and M;
            shake (bool): if True, will add perturbations to Y_tilde and M_tilde
                randomly drawn from their 2d errors (with correlation);
            sigma_scatter_M (float): the dispersion of the mass proxy.
        """

        # ======== Scattered mass proxy
        self.M_tilde = np.random.normal(self.M_tilde, sigma_scatter_M)

        # ======== Random error bars
        # ==== log Y_tilde
        if err_frac_Y is not None:
            err_log_Y_tilde = random_normal_positive(*err_frac_Y, self.n)
        else:
            err_log_Y_tilde = np.zeros(self.n)
        self.err_log_Y_tilde = err_log_Y_tilde

        # ==== log M_tilde
        if err_frac_M is not None:
            err_log_M_tilde = random_normal_positive(*err_frac_M, self.n)
        else:
            err_log_M_tilde = np.zeros(self.n)
        self.err_log_M_tilde = err_log_M_tilde

        # ==== logY-logM covariances
        if corr is not None:
            corr_logYlogM = np.random.uniform(*corr, self.n)
        else:
            corr_logYlogM = np.zeros(self.n)
        self.corr = corr_logYlogM
        covmats = [
            np.array([[dY ** 2, rho * dY * dM], [rho * dY * dM, dM ** 2]])
            for rho, dY, dM in zip(corr_logYlogM, err_log_Y_tilde, err_log_M_tilde)
        ]

        # ======== Create perturbations
        if shake:
            perturbs = np.array(
                [np.random.multivariate_normal([0, 0], covmat) for covmat in covmats]
            )
            self.log_Y_tilde += perturbs[:, 0]
            self.log_M_tilde += perturbs[:, 1]
        else:
            dlogY, dlogM = np.zeros(self.n), np.zeros(self.n)

        # Ensure that nothing was jumped below threshold
        # if hasattr(self, "log_Y_tilde_threshold"):
        #     lower = self.log_Y_tilde < self.log_Y_tilde_threshold
        #     self.log_Y_tilde[lower] = self.log_Y_tilde_threshold[lower]

        # ======== Storage
        self.Y_tilde = 10 ** self.log_Y_tilde
        if "log_Y_tilde" not in self.qts:
            self.qts.append("log_Y_tilde")
        if "err_log_Y_tilde" not in self.qts:
            self.qts.append("err_log_Y_tilde")
        self.M_tilde = 10 ** self.log_M_tilde
        if "log_M_tilde" not in self.qts:
            self.qts.append("log_M_tilde")
        if "err_log_M_tilde" not in self.qts:
            self.qts.append("err_log_M_tilde")
        if "corr" not in self.qts:
            self.qts.append("corr")

    # ============================================================================ #

    def select_from_boxes(self, boxes, numbers):
        """
        Creates a subsample from (Y_tilde - z) boxes

        Args:
            boxes (list): the boxes in which you want a sample.
                Each element must be a ``Box`` instance;
            numbers (list): how many clusters to put in each box.
                If there are not enough existing clusters, the
                box will be filled with all existing ones.
        """
        full_mask = np.ones(self.n, dtype=bool)
        self.log_Y_tilde_threshold = np.zeros(self.n)
        if "log_Y_tilde_threshold" not in self.qts:
            self.qts += ["log_Y_tilde_threshold"]
        for box, n in zip(boxes, numbers):
            eligible = np.where(box.is_in_box(self.Y_tilde, self.z))[0]
            selected = np.random.choice(
                eligible, np.min([n, eligible.size]), replace=False
            )
            for i in selected:
                full_mask[i] = False
                self.log_Y_tilde_threshold[i] = np.log10(box.y_inf)
            if eligible.size < n:
                print(
                    f"Selection: Could only take {eligible.size} clusters "
                    + f"(instead of {n}) with "
                    + f"{box.y_inf} < y < {box.y_sup} and "
                    + f"{box.z_inf} < z < {box.z_sup}."
                )

        new_cat = self.to_table()
        new_cat.remove_rows(full_mask)
        new_inst = self.from_table(new_cat, boxes=boxes)
        return new_inst

    # ============================================================================ #

    def plot_mz(self, plot_file=None, add_points=True):
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        ax.set_xlabel(r"Redshift $z$")
        ax.set_ylabel(r"Mass $M_{500}\;[{\rm M_\odot}]$")
        fkplotlib.ax_bothticks(ax)
        ax.set_ylim(2e14, 2e15)

        # ======== Right axis = E(z)^-2/3 Da^2 Y_500 [Mpc2]
        if "Y_tilde" in self.qts:
            ax2 = ax.twinx()
            ax2.set_yscale("log")
            _, lims, _ = planck_scaling_relation(
                np.ones(2), np.array(ax.get_ylim()) * u.Msun, scatter=False
            )
            ax2.set_ylim(*lims)
            ax2.set_ylabel(
                r"$E^{-2/3}(z) \, \mathcal{D}_\mathrm{A}^2 \, Y_{500}"
                + r"\; [{\rm 10^{-4} Mpc^2}]$"
            )

        # ======== Add points for clusters in self
        if add_points:
            cat = self.to_table()
            if "Y_tilde" in cat.columns:  # in SZ observable scale
                msk = np.zeros(len(cat), dtype="bool")
                for i in np.random.randint(0, len(cat), 10000):
                    msk[i] = True
                ax2.plot(
                    cat["z"][msk], cat["Y_tilde"][msk], ".", color="tab:red", alpha=0.5
                )
            else:  # in mass scale
                ax.plot(cat["z"], cat["M_500"], ".", color="tab:red", alpha=0.5)

        # ======== Add boxes
        for box in self.boxes:
            box.plot_on_ax(ax2, color="tab:red", ls="--", alpha=0.5)

        if plot_file is not None:
            fig.savefig(plot_file)
        return fig, ax


# ================================================================================ #
# ================================================================================ #


def planck_scaling_relation(z, M_500, scatter=True):
    alpha = 1.79
    log_Y_star = -0.19
    sigma_log_Y_star = 0.075 if scatter else 0.0

    M_tilde = (M_500.to("Msun") / (6e14 * u.Msun)).value
    log_M_tilde = np.log10(M_tilde)
    log_Y_tilde = np.random.normal(log_Y_star + alpha * log_M_tilde, sigma_log_Y_star)
    Y_tilde = 10 ** log_Y_tilde
    Y_500 = Y_tilde * cosmo.efunc(z) ** (2.0 / 3.0) * 1e-4 * u.Unit("Mpc2")
    return M_tilde, Y_tilde, Y_500


# ================================================================================ #


def random_normal_positive(mu, sigma, n):
    result = np.random.normal(mu, sigma, n)
    is_neg = result < 0.0
    while np.any(is_neg):
        result[is_neg] = np.random.normal(mu, sigma, np.sum(is_neg))
        is_neg = result < 0.0
    return np.array(result)


# ================================================================================ #
# ================================================================================ #
