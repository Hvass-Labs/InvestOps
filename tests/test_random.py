###############################################################################
#
# Tests for investops.random
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

from investops.random import rand_normal, rand_uniform
from investops.random import rand_corr_normal, rand_corr_uniform
from investops.check import check_corr_matrix
import numpy as np
from tests.utils import assert_array_less_equal

###############################################################################
# Settings for random tests.

# Random Number Generator.
_rng = np.random.default_rng()

# Number of random trials.
_num_trials = 100

# Min and max number of assets in each random trial.
_num_assets_min = 2
_num_assets_max = 100

###############################################################################
# rand_weights_uniform() and rand_weights_normal()


def test_rand_weights():
    """
    Test investops.random.rand_normal() and investops.random.rand_uniform().
    """
    # We only test that the random arrays have the correct shape and that
    # their elements are within the correct range. We do not check if the
    # elements are some exact values, because these will change on each run.
    # Even if we used a seed for the Random Number Generator (RNG), the
    # implementation of the RNG could change in the future, so they would
    # generate other random numbers.

    for i in range(_num_trials):
        # Random number of assets.
        num_assets = int(_rng.uniform(_num_assets_min, _num_assets_max))

        # Random min and max weights.
        min_weight = _rng.uniform(-2.0, 2.0)
        max_weight = _rng.uniform(min_weight, 3.0)

        # Random mean and std.dev.
        mean = _rng.uniform(min_weight, max_weight)
        std = _rng.uniform(0.0, 1.0)

        # Random weights. Normal-distributed.
        weights = rand_normal(rng=_rng, size=num_assets,
                              mean=mean, std=std,
                              low=min_weight, high=max_weight)

        # Check shape of the array.
        msg_wrong_shape = 'Array `weights` has wrong shape.'
        assert weights.shape == (num_assets,), msg_wrong_shape

        # Check random weights are within limits.
        assert_array_less_equal(weights, max_weight)
        assert_array_less_equal(min_weight, weights)

        # Random weights. Uniform-distributed.
        weights = rand_uniform(rng=_rng, size=num_assets,
                               low=min_weight, high=max_weight)

        # Check shape of the array.
        msg_wrong_shape = 'Array `weights` has wrong shape.'
        assert weights.shape == (num_assets,), msg_wrong_shape

        # Check random weights are within limits.
        assert_array_less_equal(weights, max_weight)
        assert_array_less_equal(min_weight, weights)


###############################################################################
# rand_corr_uniform() and rand_corr_normal()


def test_rand_corr():
    """
    Test investops.random.rand_corr_normal()
    and investops.random.rand_corr_uniform().
    """
    # We use the function `check_corr_matrix()` to check that the randomly
    # generated correlation matrices are valid.

    for i in range(_num_trials):
        # Random number of assets.
        num_assets = int(_rng.uniform(_num_assets_min, _num_assets_max))

        # Random min and max corr.
        min_corr = _rng.uniform(-1.0, 0.9)
        max_corr = _rng.uniform(min_corr, 1.0)

        # Random mean and std.dev.
        mean = _rng.uniform(min_corr, max_corr)
        std = _rng.uniform(0.0, 1.0)

        # Random weights. Normal-distributed.
        corr = rand_corr_normal(rng=_rng, num_assets=num_assets,
                                mean=mean, std=std)

        # Check the random correlation matrix is valid.
        check_corr_matrix(corr=corr)

        # Random weights. Uniform-distributed.
        weights = rand_corr_uniform(rng=_rng, num_assets=num_assets,
                                    min_corr=min_corr, max_corr=max_corr)

        # Check the random correlation matrix is valid.
        check_corr_matrix(corr=corr)


###############################################################################
