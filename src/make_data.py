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
    treatment_sds_pct: list[int],
):
    """_summary_

    Parameters
    ----------
    passes_per_treatment : int
        _description_
    tnames : list[str]
        _description_
    effect_size : list[int]
        _description_
    treatment_sds_pct : list[int]
        _description_

    Returns
    -------
    _type_
        _description_
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
                [[sd] * passes_per_treatment for sd in treatment_sds_pct]
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
    """_summary_

    Parameters
    ----------
    passes_per_treatment : int, optional
        _description_, by default 3
    pass_width : int, optional
        _description_, by default 5
    buffer : int, optional
        _description_, by default 5
    pass_length : int, optional
        _description_, by default 100
    points_per_pass : int, optional
        _description_, by default 100
    tnames : list[str], optional
        _description_, by default ["T1", "Control", "T2"]
    effect_size : list[int], optional
        _description_, by default [5, 0, 1]
    treatment_sds : list[float], optional
        _description_, by default [0.2, 0.2, 0.15]

    Returns
    -------
    pd.DataFrame
        _description_
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
    """_summary_

    Parameters
    ----------
    x : np.ndarray
        _description_
    y : np.ndarray
        _description_
    cov_fxn : callable, optional
        RBF (ExpQuad) kernel for 2D GP, by default pm.gp.cov.Exponential
    lin_cov : bool, optional
        _description_
    lin_cov_args : dict, by default {}
        Contains arguments to pass to linear GP, ie {"c": `c`}

    Returns
    -------
    np.ndarray
        _description_
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


def make_linear_pattern(x, y, c, cov_fxn=pm.gp.cov.Linear) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    x : _type_
        _description_
    y : _type_
        _description_
    c : _type_
        _description_
    cov_fxn : _type_, optional
        _description_, by default pm.gp.cov.Linear

    Returns
    -------
    np.ndarray
        _description_
    """
    return make_spatial_pattern(x, y, cov_fxn=cov_fxn, lin_cov=True, lin_cov_args={"c": c})
