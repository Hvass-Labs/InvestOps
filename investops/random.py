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
from investops.check import fix_corr_matrix

###############################################################################
# Random Weights.

def rand_weights_uniform(rng, num_assets, min_weight=0.0, max_weight=1.0):
    """
    Generate random portfolio weights using a uniform distribution.

    :param rng: `Numpy.random.Generator` object from `np.random.default_rng()`
    :param num_assets: Integer with the number of portfolio assets.
    :param min_weight: Minimum portfolio weight.
    :param max_weight: Maximum portfolio weight.
    :return: Numpy array with random weights.
    """
    return rng.uniform(low=min_weight, high=max_weight, size=num_assets)


def rand_weights_normal(rng, num_assets, mean=0.0, std=0.04,
                        min_weight=0.0, max_weight=1.0):
    """
    Generate random portfolio weights using a normal distribution that
    are also clipped between the given `min_weight` and `max_weight`.

    :param rng: `Numpy.random.Generator` object from `np.random.default_rng()`
    :param num_assets: Integer with the number of portfolio assets.
    :param mean: Mean of the normal distribution.
    :param std: Std.dev. of the normal distribution.
    :param min_weight: Minimum portfolio weight.
    :param max_weight: Maximum portfolio weight.
    :return: Numpy array with random weights.
    """
    # Generate random portfolio weights.
    weights = rng.normal(scale=std, loc=mean, size=num_assets)

    # Clip the portfolio weights.
    weights = np.clip(weights, min_weight, max_weight)

    return weights


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
