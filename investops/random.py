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

def gen_asset_names(num_assets):
    """
    Generate list of strings with default asset-names.

    :param num_assets: Integer with the number of assets.
    :return: List of strings.
    """
    return [f'Asset {i}' for i in range(num_assets)]


def gen_group_names(num_groups):
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
            names = gen_asset_names(num_assets=num_assets)
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
# Random portfolio groups.

def rand_groups(rng, num_assets, num_groups,
                max_groups_per_asset, min_groups_per_asset=0,
                asset_names=None, group_names=None):
    """
    Generate dict that maps from asset-names to lists of random group-names.

    This is intended to be used with the class `GroupConstraints`.

    :param rng: `Numpy.random.Generator` object from `np.random.default_rng()`
    :param num_assets: Integer with the number of portfolio assets.
    :param num_groups: Integer with the number of portfolio groups.
    :param max_groups_per_asset: Int with max number of groups per asset.
    :param min_groups_per_asset: Int with min number of groups per asset.
    :param asset_names:
        If list then use those asset-names, otherwise use default asset-names.
    :param group_names:
        If list then use those group-names, otherwise use default group-names.
    :return: Dict that maps asset-names to lists of group-names.
    """
    # Check number of groups is valid.
    if min_groups_per_asset > num_groups or max_groups_per_asset > num_groups \
            or min_groups_per_asset > max_groups_per_asset:
        msg = f'Arguments \'min_groups_per_asset\'={min_groups_per_asset}, ' \
              f'\'max_groups_per_asset\'={max_groups_per_asset}, and ' \
              f'\'num_groups\'={num_groups} are invalid.'
        raise ValueError(msg)

    # Asset-names.
    if not isinstance(asset_names, list):
        # Generate list of default asset-names.
        asset_names = gen_asset_names(num_assets=num_assets)
    elif len(asset_names) != num_assets:
        # The list of given asset-names has the wrong length.
        msg = f'Argument \'asset_names\' has wrong length {len(asset_names)} ' \
              f'expected {num_assets}.'
        raise ValueError(msg)

    # Group-names.
    if not isinstance(group_names, list):
        # Generate list of default group-names.
        group_names = gen_group_names(num_groups=num_groups)
    elif len(group_names) != num_groups:
        # The list of given group-names has the wrong length.
        msg = f'Argument \'group_names\' has wrong length {len(group_names)} ' \
              f'expected {num_groups}.'
        raise ValueError(msg)

    # Initialize the dict that we will build.
    asset_to_groups = dict()

    # For each asset-name in the portfolio.
    for asset_name in asset_names:
        # Random number of groups to select.
        num_groups = rng.integers(low=min_groups_per_asset,
                                  high=max_groups_per_asset + 1)

        # Random selection of group-names.
        groups = rng.choice(group_names, size=num_groups, replace=False)

        # Sort the group-names.
        groups.sort()

        # Add this asset and list of group-names to the dict.
        asset_to_groups[asset_name] = groups.tolist()

    return asset_to_groups


def rand_group_lim_uniform(rng, num_groups,
                           min_lim=0.0, max_lim=1.0, names=None):
    """
    Generate random portfolio group-limits using a uniform distribution.

    This is intended to be used with the class `GroupConstraints`.

    :param rng: `Numpy.random.Generator` object from `np.random.default_rng()`
    :param num_groups: Integer with the number of portfolio groups.
    :param min_lim: Minimum group-limit.
    :param max_lim: Maximum group-limit.
    :param names:
        If list then use those group-names, otherwise use default group-names.
    :return: Pandas Series with random group-limits.
    """
    if not isinstance(names, list):
        # Generate list of default group-names.
        names = gen_group_names(num_groups=num_groups)
    elif len(names) != num_groups:
        # The list of given group-names has the wrong length.
        msg = f'Argument \'names\' has wrong length {len(names)} ' \
              f'expected {num_groups}.'
        raise ValueError(msg)

    # Generate random group-limits.
    group_lim = rng.uniform(low=min_lim, high=max_lim, size=num_groups)

    # Convert Numpy array to Pandas Series.
    group_lim = pd.Series(data=group_lim, index=names)

    return group_lim


def rand_group_lim_normal(rng, num_groups, mean=0.0, std=0.04,
                          min_lim=0.0, max_lim=1.0, names=None):
    """
    Generate random portfolio group-limits using a normal distribution.

    This is intended to be used with the class `GroupConstraints`.

    :param rng: `Numpy.random.Generator` object from `np.random.default_rng()`
    :param num_groups: Integer with the number of portfolio groups.
    :param mean: Mean of the normal distribution.
    :param std: Std.dev. of the normal distribution.
    :param min_lim: Minimum group-limit.
    :param max_lim: Maximum group-limit.
    :param names:
        If list then use those group-names, otherwise use default group-names.
    :return: Pandas Series with random group-limits.
    """
    if not isinstance(names, list):
        # Generate list of default group-names.
        names = gen_group_names(num_groups=num_groups)
    elif len(names) != num_groups:
        # The list of given group-names has the wrong length.
        msg = f'Argument \'names\' has wrong length {len(names)} ' \
              f'expected {num_groups}.'
        raise ValueError(msg)

    # Generate random group-limits.
    group_lim = rng.normal(scale=std, loc=mean, size=num_groups)

    # Clip the group-limits.
    group_lim = np.clip(group_lim, min_lim, max_lim)

    # Convert Numpy array to Pandas Series.
    group_lim = pd.Series(data=group_lim, index=names)

    return group_lim

###############################################################################
