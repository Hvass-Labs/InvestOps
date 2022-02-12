###############################################################################
#
# Tests for investops.diversify
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

from investops.diversify import full_exposure, diversify_weights
from investops.random import rand_normal, rand_corr_normal
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from tests.utils import assert_array_less_equal

###############################################################################
# Settings for random tests.

# Random Number Generator.
_rng = np.random.default_rng()

# Number of random trials.
_num_trials = 100

# Number of assets in each random trial.
_num_assets = 100

###############################################################################
# full_exposure()


def test_full_exposure():
    """Test investops.diversify.full_exposure()."""
    # Test 1
    weights_org = np.array([0.0, 0.0, 0.0])
    corr = np.array([[1.0, 0.5, 0.2],
                     [0.5, 1.0, 0.3],
                     [0.2, 0.3, 1.0]])
    full_exp = full_exposure(weights=weights_org, corr=corr)
    assert_allclose(full_exp, [0.0, 0.0, 0.0])

    # Test 2
    weights_org = np.array([0.0, 0.5, 0.0])
    corr = np.array([[1.0, 0.5, 0.2],
                     [0.5, 1.0, 0.3],
                     [0.2, 0.3, 1.0]])
    full_exp = full_exposure(weights=weights_org, corr=corr)
    assert_allclose(full_exp, [0.0, 0.5, 0.0])

    # Test 3
    weights_org = np.array([0.3, 0.2, 0.1])
    corr = np.array([[1.0, 0.5, 0.2],
                     [0.5, 1.0, 0.3],
                     [0.2, 0.3, 1.0]])
    full_exp = full_exposure(weights=weights_org, corr=corr)
    assert_allclose(full_exp, [0.32588341, 0.23832751, 0.11401754])

    # Test 4
    weights_org = np.array([0.3, -0.2, 0.1])
    corr = np.array([[1.0, -0.5, 0.2],
                     [-0.5, 1.0, 0.3],
                     [0.2, 0.3, 1.0]])
    full_exp = full_exposure(weights=weights_org, corr=corr)
    assert_allclose(full_exp, [0.32588341, -0.23452079, 0.10583005])

    # Test 5
    weights_org = np.array([0.3, -0.2, -0.1])
    corr = np.array([[1.0, -0.5, 0.2],
                     [-0.5, 1.0, -0.3],
                     [0.2, -0.3, 1.0]])
    full_exp = full_exposure(weights=weights_org, corr=corr)
    assert_allclose(full_exp, [0.32403703, -0.23452079, -0.1])

    # Test 6
    weights_org = np.array([0.0, -0.2, -0.1])
    corr = np.array([[1.0, -0.5, 0.2],
                     [-0.5, 1.0, -0.3],
                     [0.2, -0.3, 1.0]])
    full_exp = full_exposure(weights=weights_org, corr=corr)
    assert_allclose(full_exp, [0.0, -0.2, -0.1])


def test_full_exposure_rand():
    """Test investops.diversify.full_exposure() with random data."""
    for i in range(_num_trials):
        # Random portfolio weights.
        weights_org = \
            rand_normal(rng=_rng, size=_num_assets, low=-1.0, high=1.0)

        # Random correlation matrix.
        corr = rand_corr_normal(rng=_rng, num_assets=_num_assets)

        # Calculate the Full Exposure.
        full_exp = full_exposure(weights=weights_org, corr=corr)

        # Assert abs(Weight[i]) <= abs(FullExp[i])
        assert_array_less_equal(np.abs(weights_org), np.abs(full_exp))

        # Assert sign(Weight[i]) == sign(FullExp[i])
        assert_array_equal(np.sign(full_exp), np.sign(weights_org))


###############################################################################
# diversify_weights


def test_diversify_weights():
    """Test investops.diversify.diversify_weights()."""
    # Test 1
    weights_org = np.array([0.0, 0.0, 0.0])
    corr = np.array([[1.0, 0.5, 0.2],
                     [0.5, 1.0, 0.3],
                     [0.2, 0.3, 1.0]])
    weights_new = diversify_weights(weights_org=weights_org, corr=corr)
    assert_allclose(weights_new, [0.0, 0.0, 0.0])

    # Test 2
    weights_org = np.array([0.1, 0.0, 0.0])
    corr = np.array([[1.0, 0.5, 0.2],
                     [0.5, 1.0, 0.3],
                     [0.2, 0.3, 1.0]])
    weights_new = diversify_weights(weights_org=weights_org, corr=corr)
    assert_allclose(weights_new, [0.1, 0.0, 0.0])

    # Test 3
    weights_org = np.array([0.1, 0.2, 0.3])
    corr = np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])
    weights_new = diversify_weights(weights_org=weights_org, corr=corr)
    assert_allclose(weights_new, [0.1, 0.2, 0.3])

    # Test 4
    weights_org = np.array([0.1, 0.2, 0.3])
    corr = np.array([[1.0, 0.5, 0.2],
                     [0.5, 1.0, 0.3],
                     [0.2, 0.3, 1.0]])
    weights_new = diversify_weights(weights_org=weights_org, corr=corr)
    assert_allclose(weights_new, [0.07589274, 0.17863085, 0.29059378])

    # Test 5
    weights_org = np.array([0.1, 0.2, -0.3])
    corr = np.array([[1.0, 0.5, 0.2],
                     [0.5, 1.0, -0.3],
                     [0.2, -0.3, 1.0]])
    weights_new = diversify_weights(weights_org=weights_org, corr=corr)
    assert_allclose(weights_new, [0.08023448,  0.17813341, -0.29209408])

    # Test 6
    weights_org = np.array([-0.1, 0.2, -0.3])
    weights_guess = np.array([1234.5, -678.9, 876.543])
    corr = np.array([[1.0, -0.5, 0.2],
                     [-0.5, 1.0, -0.3],
                     [0.2, -0.3, 1.0]])
    weights_new = diversify_weights(weights_org=weights_org, corr=corr,
                                    weights_guess=weights_guess)
    assert_allclose(weights_new, [-0.07581472, 0.178671, -0.29059834])


def test_diversify_weights_rand():
    """Test investops.diversify.diversify_weights() with random data."""
    for i in range(_num_trials):
        # Random portfolio weights.
        weights_org = \
            rand_normal(rng=_rng, size=_num_assets, low=-1.0, high=1.0)

        # Random correlation matrix.
        corr = rand_corr_normal(rng=_rng, num_assets=_num_assets)

        # Calculate the diversified weights.
        weights_new = diversify_weights(weights_org=weights_org, corr=corr)

        # Assert abs(weights_new[i]) <= abs(weights_org[i])
        assert_array_less_equal(np.abs(weights_new), np.abs(weights_org))

        # Assert sign(weights_new[i]) == sign(weights_org[i])
        assert_array_equal(np.sign(weights_new), np.sign(weights_org))


###############################################################################
