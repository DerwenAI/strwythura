#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Numerical utilities used for optimization.
see copyright/license https://github.com/DerwenAI/strwythura/README.md
"""

import math
import typing

import numpy as np
import pandas as pd


def calc_quantile_bins (
    num_rows: int,
    *,
    amplitude: int = 4,
    ) -> np.ndarray:
    """
Calculate the bins to use for a quantile stripe,
using [`numpy.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)

    num_rows:
number of rows in the target dataframe

    returns:
calculated bins, as a `numpy.ndarray`
    """
    granularity = max(round(math.log(num_rows) * amplitude), 1)

    return np.linspace(
        0,
        1,
        num = granularity,
        endpoint = True,
    )


def stripe_column (
    values: list,
    bins: int,
    ) -> np.ndarray:
    """
Stripe a column in a dataframe, by interpolating quantiles into a set of discrete indexes.

    values:
list of values to stripe

    bins:
quantile bins; see [`calc_quantile_bins()`](#calc_quantile_bins-function)

    returns:
the striped column values, as a `numpy.ndarray`
    """
    s = pd.Series(values)
    q = s.quantile(bins, interpolation = "nearest")

    try:
        stripe = np.digitize(values, q) - 1
        return stripe
    except ValueError as ex:
        # should never happen?
        print("ValueError:", str(ex), values, s, q, bins)
        raise


def root_mean_square (
    values: typing.List[ float ]
    ) -> float:
    """
Calculate the [*root mean square*](https://mathworld.wolfram.com/Root-Mean-Square.html)
of the values in the given list.

    values:
list of values to use in the RMS calculation

    returns:
RMS metric as a float
    """
    s: float = sum(map(lambda x: float(x) ** 2.0, values))
    n: float = float(len(values))

    return math.sqrt(s / n)
