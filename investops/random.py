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

def _get_size(size=None, index=None, columns=None):
    """
    Determine the desired array-size from the given arguments:
    - If `index` is None, return the given `size.
    - If `index` and `columns` are not None, return tuple with their lengths.
    - If `index` is not None, return int with its length.

    :param size: Int or tuple of ints.
    :param index: List of strings with index-names.
    :param columns: List of strings with column-names.
    :return: Int or tuple with desired array-size.
    """
    # Check the arguments are valid.
    if size is None and index is None:
        msg = 'Arguments \'size\' and \'index\' cannot both be None.'
        raise ValueError(msg)

    # Determine the desired array-size.
    if index is not None:
        if columns is not None:
            size = (len(index), len(columns))
        else:
            size = len(index)

    return size


def _named_values(values, index=None, columns=None):
    """
    Convert a Numpy array to either a Pandas Series or DataFrame,
    depending on whether the `index` and `columns` are given.

    :param values: Numpy array.
    :param index: List of strings for the index-names.
    :param columns: List of strings for the column-names.
    :return:
        - Numpy array if `index` is None.
        - Pandas Series if `index is not None.
        - Pandas DataFrame if both `index` and `columns` are not None.
    """
    if index is not None:
        if columns is not None:
            # Convert to Pandas DataFrame.
            values = pd.DataFrame(data=values, index=index, columns=columns)
        else:
            # Convert to Pandas Series.
            values = pd.Series(data=values, index=index)

    return values


###############################################################################
# Random numbers.

def rand_uniform(rng, low=0.0, high=1.0, size=None, index=None, columns=None):
    """
    Generate random numbers from a uniform distribution between the given
    `low` and `high`. The output is a Numpy array if only `size` is provided,
    and a Pandas Series if `index` is provided, and a Pandas DataFrame if
    both `index` and `columns` are provided.

    :param rng:
        `Numpy.random.Generator` object from `np.random.default_rng()`

    :param size:
        Int or tuple of ints with the size of the array to generate.
        This is only used if the argument `index` is None.

    :param mean:
        Float with the mean of the normal distribution.

    :param std:
        Float with the std.dev. of the normal distribution.

    :param low:
        Float with lower limit for the random numbers.

    :param high:
        Float with upper limit for the random numbers.

    :param index:
        List of strings with the index-names.

    :param columns:
        List of strings with the column-names.

    :return:
        - Numpy array if `index` is None.
        - Pandas Series if `index` is not None.
        - Pandas DataFrame if both `index` and `columns` are not None.
    """
    # Determine the size of the array we must generate.
    size = _get_size(size=size, index=index, columns=columns)

    # Generate Numpy array of random numbers.
    values = rng.uniform(low=low, high=high, size=size)

    # If necessary convert to Pandas data before returning the random data.
    return _named_values(values=values, index=index, columns=columns)


def rand_normal(rng, mean=0.0, std=0.04, low=0.0, high=1.0,
                size=None, index=None, columns=None):
    """
    Generate random numbers from a normal distribution that are also clipped
    between the given `low` and `high`. The output is a Numpy array if only
    `size` is provided, and a Pandas Series if `index` is provided, and a
    Pandas DataFrame if both `index` and `columns` are provided.

    :param rng:
        `Numpy.random.Generator` object from `np.random.default_rng()`

    :param size:
        Int or tuple of ints with the size of the array to generate.
        This is only used if the argument `index` is None.

    :param mean:
        Float with the mean of the normal distribution.

    :param std:
        Float with the std.dev. of the normal distribution.

    :param low:
        Float with lower limit for the random numbers.

    :param high:
        Float with upper limit for the random numbers.

    :param index:
        List of strings with the index-names.

    :param columns:
        List of strings with the column-names.

    :return:
        - Numpy array if `index` is None.
        - Pandas Series if `index` is not None.
        - Pandas DataFrame if both `index` and `columns` are not None.
    """
    # Determine the size of the array we must generate.
    size = _get_size(size=size, index=index, columns=columns)

    # Generate Numpy array of random numbers.
    values = rng.normal(scale=std, loc=mean, size=size)

    # Clip the values?
    if low is not None and high is not None:
        values = np.clip(values, low, high)

    # If necessary convert to Pandas data before returning the random data.
    return _named_values(values=values, index=index, columns=columns)

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

def rand_where(rng, x, y, prob):
    """
    Randomly return elements chosen from `x` or `y` depending on probability.

    :param rng: `Numpy.random.Generator` object from `np.random.default_rng()`
    :param x: Numpy or Pandas array which is copied and NOT updated inplace.
    :param y: Other Numpy array or scalar value.
    :param prob:
        Float in [0,1] with probability of using values from `x` or `y`.
        A `prob` value of 0.0 always selects values from `x`, and
        a `prob` value of 1.0 always selects value from `y`.
    :return: A new Numpy or Pandas array of the same shape as `x`.
    """
    # Array with random values between [0,1].
    p = rng.uniform(size=x.shape)

    # Create a copy of the array x where random values are set to y.
    # This is always a Numpy array even if x was Pandas data.
    x_new = np.where(prob < p, x, y)

    # Convert result back to Pandas?
    if isinstance(x, pd.Series):
        # Convert result to Pandas Series.
        x_new = pd.Series(data=x_new, index=x.index)
    elif isinstance(x, pd.DataFrame):
        # Convert result to Pandas DataFrame.
        x_new = pd.DataFrame(data=x_new, index=x.index, columns=x.columns)

    return x_new


def rand_zero(rng, x, prob):
    """
    Randomly set values of a Numpy array to zero according to a probability.

    :param rng: `Numpy.random.Generator` object from `np.random.default_rng()`
    :param x: Numpy array which is copied and NOT updated inplace.
    :param prob: Float with the probability that an element is set to zero.
    :return: A new Numpy array of the same shape as `x`.
    """
    return rand_where(rng=rng, x=x, y=0, prob=prob)

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

###############################################################################
