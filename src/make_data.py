import itertools
import geopandas as gpd
import numpy as np
import pandas as pd
import pymc as pm
import random


def make_df_meta(
    passes_per_treatment: int,
    tnames: list[str],
    effect_size: list[int],
    treatment_sds: list[int],
):
    """Helper function to create df_meta

    Parameters
    ----------
    passes_per_treatment : int
        The number of passes per treatment
    tnames : list[str]
        Treatment names
    effect_size : list[int]
        Effect sizes
    treatment_sds : list[int]
        Absolute standard deviations for each treatment

    Returns
    -------
    pd.DataFrame
        Contains "treatmentname", "pass_no", "effect_size", and "sd"
    """
    return pd.DataFrame(
        {
            "treatmentname": itertools.chain.from_iterable(
                [[tname] * passes_per_treatment for tname in tnames]
            ),
            "pass_no": list(range(1, len(tnames) * passes_per_treatment + 1, 1)),
            "effect_size": itertools.chain.from_iterable(
                [[esize] * passes_per_treatment for esize in effect_size]
            ),
            "sd": itertools.chain.from_iterable(
                [[sd] * passes_per_treatment for sd in treatment_sds]
            ),
        }
    )


def make_dummy_block(
    passes_per_treatment: int = 3,
    pass_width: int = 5,
    buffer: int = 5,
    pass_length: int = 100,
    points_per_pass: int = 100,
    tnames: list[str] = ["T1", "Control", "T2"],
    effect_size: list[int] = [5, 0, 1],
    treatment_sds: list[float] = [1, 0.2, 0.15],
) -> pd.DataFrame:
    """Make dummy data for an experiment block

    Parameters
    ----------
    passes_per_treatment : int, optional
        The number of passes per treatment, by default 3
    pass_width : int, optional
        The width of a single pass, by default 5
    buffer : int, optional
        The distance between treatments, by default 5
    pass_length : int, optional
        The length of a pass, by default 100
    points_per_pass : int, optional
        The number of points in a pass, by default 100
    tnames : list[str], optional
        Treatment names, by default ["T1", "Control", "T2"]
    effect_size : list[int], optional
        Effect sizes, by default [5, 0, 1]
    treatment_sds : list[float], optional
        Absolute standard deviations for each treatment, by default [0.2, 0.2, 0.15]

    Returns
    -------
    pd.DataFrame
        Contains "treatmentname", "x_lon", "y_lat", and "value"
    """
    # first create df with meta-data
    df_meta = make_df_meta(passes_per_treatment, tnames, effect_size, treatment_sds)
    random.seed(10)
    # create df_field
    groupby_cols = ["treatmentname", "pass_no", "effect_size", "sd"]
    x_interval = pass_width + buffer
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "treatmentname": [tname] * points_per_pass,
                    "x_lon": [pass_no * x_interval] * points_per_pass,
                    "y_lat": np.sort([random.uniform(0, pass_length) for _ in range(points_per_pass)]),
                    "value": [np.random.normal(esize, sd) for _ in range(points_per_pass)],
                }
            ) for (tname, pass_no, esize, sd), _ in df_meta.groupby(groupby_cols)
        ],
        ignore_index=True
    )


def make_spatial_pattern(
    x: np.ndarray,
    y: np.ndarray,
    cov_fxn: callable = pm.gp.cov.Exponential,
    lin_cov=False,
    lin_cov_args={},
) -> np.ndarray:
    """Make a spatial pattern

    Parameters
    ----------
    x : np.ndarray
        Values in the x-direction
    y : np.ndarray
        Values in the y-direction
    cov_fxn : callable, optional
        RBF (ExpQuad) kernel for 2D GP, by default pm.gp.cov.Exponential
    lin_cov : bool, optional
        Flag to denote whether to use a linear covariance fxn
    lin_cov_args : dict, by default {}
        Contains arguments to pass to linear GP, ie {"c": `c`}

    Returns
    -------
    np.ndarray
        Values for the spatial pattern
    """
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T
    c = lin_cov_args.get("c", 1) if lin_cov else None
    with pm.Model() as model:
        # Define a covariance function (kernel)
        ℓ = pm.Gamma("ℓ", alpha=2, beta=1)  # Lengthscale parameter
        cov_func = cov_fxn(2, ls=ℓ) if not lin_cov else cov_fxn(2, c)
        
        # Define the GP with no observed data (latent GP)
        gp = pm.gp.Latent(cov_func=cov_func)
        
        # GP prior
        f = gp.prior("f", X=coords)

        #Sample from the GP
        trace = pm.sample_prior_predictive(draws=1)
    return trace.prior["f"].values.reshape(len(x), len(y))


def make_linear_pattern(x: np.ndarray, y: np.ndarray, c: int, cov_fxn=pm.gp.cov.Linear) -> np.ndarray:
    """Make a linear spatial pattern

    Parameters
    ----------
    x : np.ndarray
        Values in the x-direction
    y : np.ndarray
        Values in the y-direction
    c : int
        The constant, `c`,  in the linear kernel, k(x, x') = (x - c)(x' - c)
    cov_fxn : Covariance object, optional
        The linear kernel object, by default pm.gp.cov.Linear

    Returns
    -------
    np.ndarray
        Values for the spatial pattern
    """
    return make_spatial_pattern(x, y, cov_fxn=cov_fxn, lin_cov=True, lin_cov_args={"c": c})
