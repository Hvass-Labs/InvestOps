###############################################################################
#
# Functions for generating random portfolio weights and correlation matrices.
#
###############################################################################
#
# This file is part of InvestOps:
#
# https://github.com/Hvass-Labs/InvestOps
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2021 by Magnus Erik Hvass Pedersen
#
###############################################################################

import numpy as np
import pandas as pd
from investops.check import fix_corr_matrix

###############################################################################
# Generate default names.

def gen_names_assets(num_assets):
    """
    Generate list of strings with default asset-names.

    :param num_assets: Integer with the number of assets.
    :return: List of strings.
    """
    return [f'Asset {i}' for i in range(num_assets)]


def gen_names_groups(num_groups):
    """
    Generate list of strings with default group-names.

    :param num_groups: Integer with the number of groups.
    :return: List of strings.
    """
    return [f'Group {i}' for i in range(num_groups)]

###############################################################################
# Helper-functions.

def _named_weights(weights, num_assets, names):
    """
    If `names` is not `None` then convert `weights` to a Pandas Series
    either using the given list of asset-names, or generate default names.

    :param weights: Numpy array with portfolio weights.
    :param num_assets: Integer with the number of assets.
    :param names:
        If `None` then don't use names for the weights.
        If `True` then generate default names for the asset-weights.
        If list then use those names for the asset-weights.
    :return:
        If `names` is `None` then a Numpy array, otherwise a Pandas Series.
    """
    # Use names for the weights?
    if names is not None:
        if names is True:
            # Use default asset-names.
            names = gen_names_assets(num_assets=num_assets)
        elif isinstance(names, list):
            # Use the list of names as-is, assuming they are all strings.
            # Check the number of names is correct.
            if len(names) != num_assets:
                msg = f'Argument \'names\' has wrong length {len(names)} ' \
                      f'expected {num_assets}.'
                raise ValueError(msg)

        # Convert from Numpy array to Pandas Series.
        weights = pd.Series(data=weights, index=names)

    return weights

###############################################################################
# Random Weights.

def rand_weights_uniform(rng, num_assets,
                         min_weight=0.0, max_weight=1.0, names=None):
    """
    Generate random portfolio weights using a uniform distribution.

    :param rng: `Numpy.random.Generator` object from `np.random.default_rng()`
    :param num_assets: Integer with the number of portfolio assets.
    :param min_weight: Minimum portfolio weight.
    :param max_weight: Maximum portfolio weight.
    :param names:
        If `None` then don't use names for the weights.
        If `True` then generate default names for the asset-weights.
        If list then use those names for the asset-weights.
    :return:
        If `names` is `None` then return a Numpy array with random weights.
        Otherwise return a Pandas Series with named random weights.
    """
    weights = rng.uniform(low=min_weight, high=max_weight, size=num_assets)
    return _named_weights(weights=weights, num_assets=num_assets, names=names)


def rand_weights_normal(rng, num_assets, mean=0.0, std=0.04,
                        min_weight=0.0, max_weight=1.0, names=None):
    """
    Generate random portfolio weights using a normal distribution that
    are also clipped between the given `min_weight` and `max_weight`.

    :param rng: `Numpy.random.Generator` object from `np.random.default_rng()`
    :param num_assets: Integer with the number of portfolio assets.
    :param mean: Mean of the normal distribution.
    :param std: Std.dev. of the normal distribution.
    :param min_weight: Minimum portfolio weight.
    :param max_weight: Maximum portfolio weight.
    :param names:
        If `None` then don't use names for the weights.
        If `True` then generate default names for the asset-weights.
        If list then use those names for the asset-weights.
    :return:
        If `names` is `None` then return a Numpy array with random weights.
        Otherwise return a Pandas Series with named random weights.
    """
    # Generate random portfolio weights.
    weights = rng.normal(scale=std, loc=mean, size=num_assets)

    # Clip the portfolio weights.
    weights = np.clip(weights, min_weight, max_weight)

    return _named_weights(weights=weights, num_assets=num_assets, names=names)

###############################################################################
# Random correlation matrices.

def rand_corr_uniform(rng, num_assets, min_corr=-1.0, max_corr=1.0):
    """
    Generate a random correlation matrix using a uniform distribution.
    It is symmetrical with random elements between `min_corr` and `max_corr`
    (also limited between -1 and 1), and the diagonal elements are all 1.

    :param rng: `Numpy.random.Generator` object from `np.random.default_rng()`
    :param num_assets: Integer for the number of portfolio assets.
    :param min_corr: Minimum correlation coefficient.
    :param max_corr: Maximum correlation coefficient.
    :return: Numpy matrix with random correlation coefficients.
    """
    # Generate random matrix.
    corr = rng.uniform(low=min_corr, high=max_corr,
                       size=(num_assets, num_assets))

    # Ensure it is a valid correlation matrix. Repair the data inplace.
    fix_corr_matrix(corr, copy=False)

    return corr


def rand_corr_normal(rng, num_assets, mean=0.0, std=0.2):
    """
    Generate a random correlation matrix using a normal distribution.
    It is symmetrical with random elements limited between -1 and 1,
    and the diagonal elements are all 1.

    :param rng: `Numpy.random.Generator` object from `np.random.default_rng()`
    :param num_assets: Integer for the number of portfolio assets.
    :param mean: Mean of the normal distribution.
    :param std: Std.dev. of the normal distribution.
    :return: Numpy matrix with random correlation coefficients.
    """
    # Generate random matrix.
    corr = rng.normal(scale=std, loc=mean, size=(num_assets, num_assets))

    # Ensure it is a valid correlation matrix. Repair the data inplace.
    fix_corr_matrix(corr, copy=False)

    return corr

###############################################################################

def rand_zero(rng, a, prob):
    """
    Randomly set values of a Numpy array to zero according to a probability.

    :param rng: `Numpy.random.Generator` object from `np.random.default_rng()`
    :param a: Numpy array which is copied and NOT updated inplace.
    :param prob: Float with the probability that an element is set to zero.
    :return: A new Numpy array of the same shame as `a`.
    """
    # Array with random values between [0,1].
    p = rng.uniform(size=a.shape)

    # Create a copy of the input array where random values are set to zero.
    return np.where(p < prob, 0, a)

###############################################################################
