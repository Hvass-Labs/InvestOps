###############################################################################
#
# Tests for investops.group_constraints
#
###############################################################################
#
# This file is part of InvestOps:
#
# https://github.com/Hvass-Labs/InvestOps
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2022 by Magnus Erik Hvass Pedersen
#
###############################################################################

from investops.group_constraints import GroupConstraints
from investops.random import (rand_normal, rand_uniform, rand_where,
                              rand_groups, gen_asset_names, gen_group_names)
import numpy as np
import pandas as pd

from pandas.testing import assert_series_equal
from numpy.testing import assert_array_equal
from tests.utils import assert_array_less_equal

###############################################################################
# Settings for random tests.

# Random Number Generator.
_rng = np.random.default_rng()

# Number of random trials.
# This is large to give a better chance of finding rare problems.
_num_trials = 1000

# Number of assets in each random trial.
_min_num_assets = 2
_max_num_assets = 1000

# Number of groups in each random trial.
_min_num_groups = 1
_max_num_groups = 100

###############################################################################
# GroupConstraints

def test_GroupConstraints():
    """Test investops.group_constraints.GroupConstraints."""

    # Asset names.
    ASSET_A = 'Asset A'
    ASSET_B = 'Asset B'
    ASSET_C = 'Asset C'
    ASSET_D = 'Asset D'

    # All asset names.
    asset_names = [ASSET_A, ASSET_B, ASSET_C, ASSET_D]

    # Group names.
    GROUP_A = 'Group A'
    GROUP_B = 'Group B'
    GROUP_C = 'Group C'

    # All group names.
    group_names = [GROUP_A, GROUP_B, GROUP_C]

    # Dict mapping from asset-name to list of group-names.
    asset_to_groups = \
    {
        ASSET_A: [GROUP_A],
        ASSET_B: [GROUP_A, GROUP_B],
        ASSET_C: [GROUP_B, GROUP_C],
        ASSET_D: [GROUP_C],
    }

    # Error tolerance in these tests.
    atol = 1e-3

    # Test 1
    group_lim_pos = pd.Series({GROUP_A: 0.05, GROUP_B: 0.1, GROUP_C: 0.2})
    group_lim_neg = None
    grp = GroupConstraints(asset_to_groups=asset_to_groups,
                           group_lim_pos=group_lim_pos,
                           group_lim_neg=group_lim_neg,
                           asset_names=asset_names, group_names=group_names)
    weights_org = pd.Series({ASSET_A: 0.05, ASSET_B: 0.1,
                             ASSET_C: 0.15, ASSET_D: 0.2})
    weights_test = pd.Series({ASSET_A: 0.016667, ASSET_B: 0.033333,
                              ASSET_C: 0.065788, ASSET_D: 0.134212})
    weights_new = grp.constrain_weights(weights_org=weights_org)
    assert_series_equal(weights_new, weights_test, atol=atol)

    # Test 2
    group_lim_pos = pd.Series({GROUP_A: 0.05, GROUP_B: 0.1, GROUP_C: np.inf})
    group_lim_neg = None
    grp = GroupConstraints(asset_to_groups=asset_to_groups,
                           group_lim_pos=group_lim_pos,
                           group_lim_neg=group_lim_neg,
                           asset_names=asset_names, group_names=group_names)
    weights_org = pd.Series({ASSET_A: 0.05, ASSET_B: 0.1,
                             ASSET_C: 0.15, ASSET_D: 0.2})
    weights_test = pd.Series({ASSET_A: 0.016667, ASSET_B: 0.033333,
                              ASSET_C: 0.066666, ASSET_D: 0.2})
    weights_new = grp.constrain_weights(weights_org=weights_org)
    assert_series_equal(weights_new, weights_test, atol=atol)

    # Test 3
    group_lim_pos = pd.Series({GROUP_A: 0.05, GROUP_B: 0.1, GROUP_C: np.inf})
    group_lim_neg = pd.Series({GROUP_A: -np.inf, GROUP_B: -0.08, GROUP_C: -0.2})
    grp = GroupConstraints(asset_to_groups=asset_to_groups,
                           group_lim_pos=group_lim_pos,
                           group_lim_neg=group_lim_neg,
                           asset_names=asset_names, group_names=group_names)
    weights_org = pd.Series({ASSET_A: 0.05, ASSET_B: 0.1,
                             ASSET_C: -0.15, ASSET_D: -0.2})
    weights_test = pd.Series({ASSET_A: 0.016667, ASSET_B: 0.033333,
                              ASSET_C: -0.08, ASSET_D: -0.12})
    weights_new = grp.constrain_weights(weights_org=weights_org)
    assert_series_equal(weights_new, weights_test, atol=atol)


def test_GroupConstraints_rand():
    """Test investops.group_constraints.GroupConstraints with random data."""
    for i in range(_num_trials):
        # Random configuration.
        num_assets = _rng.integers(low=_min_num_assets, high=_max_num_assets+1)
        num_groups = _rng.integers(low=_min_num_groups, high=_max_num_groups+1)
        min_groups_per_asset = _rng.integers(low=0, high=num_groups)
        max_groups_per_asset = _rng.integers(low=min_groups_per_asset+1,
                                             high=num_groups+1)

        # Default asset and group names.
        asset_names = gen_asset_names(num_assets=num_assets)
        group_names = gen_group_names(num_groups=num_groups)

        # Random portfolio weights using default asset-names.
        weights_org = rand_normal(rng=_rng, size=num_assets, index=asset_names,
                                  low=-1.0, high=1.0)

        # Dict with random mappings from asset-names to lists of group-names.
        asset_to_groups = \
            rand_groups(rng=_rng, num_assets=num_assets, num_groups=num_groups,
                        min_groups_per_asset=min_groups_per_asset,
                        max_groups_per_asset=max_groups_per_asset,
                        asset_names=asset_names, group_names=group_names)

        # Random POSITIVE group-limits. These should not be zero!
        group_lim_pos = \
            rand_uniform(rng=_rng, index=group_names, low=0.01, high=0.1)

        # Random NEGATIVE group-limits. These should not be zero!
        group_lim_neg = \
            rand_uniform(rng=_rng, index=group_names, low=-0.1, high=-0.01)

        # Randomly set some of the group-limits to infinity.
        prob = 0.05
        group_lim_pos = \
            rand_where(rng=_rng, x=group_lim_pos, y=np.inf, prob=prob)
        group_lim_neg = \
            rand_where(rng=_rng, x=group_lim_neg, y=-np.inf, prob=prob)

        # Create the solver for portfolio group constraints.
        grp = GroupConstraints(asset_to_groups=asset_to_groups,
                               group_lim_pos=group_lim_pos,
                               group_lim_neg=group_lim_neg,
                               asset_names=asset_names,
                               group_names=group_names)

        # Calculate the constrained portfolio weights.
        weights_new = grp.constrain_weights(weights_org=weights_org)

        # Assert the Pandas indices are correct.
        assert_array_equal(weights_new.index, weights_org.index)

        # Assert weights have only decreased in magnitude.
        assert_array_less_equal(np.abs(weights_new), np.abs(weights_org))

        # Assert weight-signs are correct.
        assert_array_equal(np.sign(weights_new), np.sign(weights_org))

        # Calculate the group-ratios.
        group_ratios_pos, group_ratios_neg = \
            grp.group_ratios(weights=weights_new)

        # Assert group-ratios are valid. Allow for small error.
        tol = 1e-6
        assert_array_less_equal(1.0 - tol, group_ratios_pos)
        assert_array_less_equal(1.0 - tol, group_ratios_neg)

###############################################################################
