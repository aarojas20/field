import itertools
import geopandas as gpd
import numpy as np
import pandas as pd
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
    treatment_sds_pct: list[float] = [0.2, 0.2, 0.15],
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
    treatment_sds_pct : list[float], optional
        _description_, by default [0.2, 0.2, 0.15]

    Returns
    -------
    pd.DataFrame
        _description_
    """
    # first create df with meta-data
    df_meta = make_df_meta(passes_per_treatment, tnames, effect_size, treatment_sds_pct)
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
                    "value": [np.random.normal(esize, sd_pct * esize) for _ in range(points_per_pass)],
                }
            ) for (tname, pass_no, esize, sd_pct), _ in df_meta.groupby(groupby_cols)
        ],
        ignore_index=True
    )
