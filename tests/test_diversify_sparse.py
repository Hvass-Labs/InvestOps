###############################################################################
#
# Tests for investops.diversify_sparse
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

from investops.diversify import full_exposure, diversify_weights
from investops.diversify_sparse import (full_exposure_sparse,
                                        diversify_weights_sparse)
from investops.random import rand_weights_normal, rand_corr_normal, rand_zero
from investops.check import fix_corr_matrix
from investops.sparse import matrix_to_sparse
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
# full_exposure_sparse()


def test_full_exposure_sparse():
    """Test investops.diversify.full_exposure_sparse()."""
    # Test 1
    weights_org = np.array([0.1, 0.2, 0.3])
    corr = np.array([[1.0, 0.0, 0.2],
                     [0.0, 1.0, 0.3],
                     [0.2, 0.3, 1.0]])
    corr_triu = np.triu(corr, k=1)
    corr_i, corr_j, corr_coef = matrix_to_sparse(mat=corr_triu)
    full_exp = full_exposure_sparse(weights=weights_org, corr_i=corr_i,
                                    corr_j=corr_j, corr_coef=corr_coef)
    assert_allclose(full_exp, [0.10583005, 0.21307276, 0.31080541])

    # Test 2
    weights_org = np.array([-0.1, 0.2, -0.3])
    corr = np.array([[1.0, 0.0, -0.2],
                     [0.0, 1.0, -0.3],
                     [-0.2, -0.3, 1.0]])
    corr_triu = np.triu(corr, k=1)
    corr_i, corr_j, corr_coef = matrix_to_sparse(mat=corr_triu)
    full_exp = full_exposure_sparse(weights=weights_org, corr_i=corr_i,
                                    corr_j=corr_j, corr_coef=corr_coef)
    assert_allclose(full_exp, [-0.1, 0.21307276, -0.3088689])


def test_full_exposure_sparse_rand():
    """Test investops.diversify.full_exposure_sparse() with random data."""
    for i in range(_num_trials):
        # Random portfolio weights.
        weights_org = rand_weights_normal(rng=_rng, num_assets=_num_assets,
                                          min_weight=-1.0, max_weight=1.0)

        # Random correlation matrix.
        corr_dense = rand_corr_normal(rng=_rng, num_assets=_num_assets)

        # Random sparsity between 0 and 1.
        prob_sparse = _rng.uniform()

        # Randomly set some of the matrix elements to zero.
        corr_dense = rand_zero(rng=_rng, x=corr_dense, prob=prob_sparse)

        # Make the correlation matrix symmetrical again.
        corr_dense += corr_dense.T

        # Ensure it is a valid correlation matrix.
        fix_corr_matrix(corr=corr_dense, copy=False)

        # Convert to sparse correlation matrix.
        corr_triu = np.triu(corr_dense, k=1)
        corr_i, corr_j, corr_coef = matrix_to_sparse(mat=corr_triu)

        # Calculate the Full Exposure using the DENSE algorithm.
        full_exp_dense = full_exposure(weights=weights_org, corr=corr_dense)

        # Calculate the Full Exposure using the SPARSE algorithm.
        full_exp_sparse = full_exposure_sparse(weights=weights_org,
                            corr_i=corr_i, corr_j=corr_j, corr_coef=corr_coef)

        # Assert abs(Weight[i]) <= abs(FullExp[i])
        assert_array_less_equal(np.abs(weights_org), np.abs(full_exp_sparse))

        # Assert sign(Weight[i]) == sign(FullExp[i])
        assert_array_equal(np.sign(full_exp_sparse), np.sign(weights_org))

        # Asset full_exp_sparse == full_exp_dense
        assert_allclose(full_exp_sparse, full_exp_dense)


###############################################################################
# diversify_weights_sparse()


def test_diversify_weights_sparse():
    """Test investops.diversify.diversify_weights_sparse()."""
    # Test 1
    weights_org = np.array([0.1, 0.2, 0.3])
    corr = np.array([[1.0, 0.0, 0.2],
                     [0.0, 1.0, 0.3],
                     [0.2, 0.3, 1.0]])
    corr_triu = np.triu(corr, k=1)
    corr_i, corr_j, corr_coef = matrix_to_sparse(mat=corr_triu)
    weights_new = diversify_weights_sparse(weights_org=weights_org,
                           corr_i=corr_i, corr_j=corr_j, corr_coef=corr_coef)
    assert_allclose(weights_new, [0.09438243, 0.18741386, 0.28983142])

    # Test 2
    weights_org = np.array([-0.1, 0.2, -0.3])
    corr = np.array([[1.0, 0.0, -0.2],
                     [0.0, 1.0, -0.3],
                     [-0.2, -0.3, 1.0]])
    corr_triu = np.triu(corr, k=1)
    corr_i, corr_j, corr_coef = matrix_to_sparse(mat=corr_triu)
    weights_new = diversify_weights_sparse(weights_org=weights_org,
                           corr_i=corr_i, corr_j=corr_j, corr_coef=corr_coef)
    assert_allclose(weights_new, [-0.1, 0.18734228, -0.29166328])


def test_diversify_weights_sparse_rand():
    """Test investops.diversify.diversify_weights_sparse() with random data."""
    for i in range(_num_trials):
        # Random portfolio weights.
        weights_org = rand_weights_normal(rng=_rng, num_assets=_num_assets,
                                          min_weight=-1.0, max_weight=1.0)

        # Random correlation matrix.
        corr_dense = rand_corr_normal(rng=_rng, num_assets=_num_assets)

        # Random sparsity between 0 and 1.
        prob_sparse = _rng.uniform()

        # Randomly set some of the matrix elements to zero.
        corr_dense = rand_zero(rng=_rng, x=corr_dense, prob=prob_sparse)

        # Make the correlation matrix symmetrical again.
        corr_dense += corr_dense.T

        # Ensure it is a valid correlation matrix.
        fix_corr_matrix(corr=corr_dense, copy=False)

        # Convert to sparse correlation matrix.
        corr_triu = np.triu(corr_dense, k=1)
        corr_i, corr_j, corr_coef = matrix_to_sparse(mat=corr_triu)

        # Calculate the diversified weights using the DENSE algorithm.
        weights_new_dense = diversify_weights(weights_org=weights_org,
                                              corr=corr_dense)

        # Calculate the diversified weights using the SPARSE algorithm.
        weights_new_sparse = \
            diversify_weights_sparse(weights_org=weights_org,
                            corr_i=corr_i, corr_j=corr_j, corr_coef=corr_coef)

        # Assert abs(weights_new_sparse[i]) <= abs(weights_org[i])
        assert_array_less_equal(np.abs(weights_new_sparse),
                                np.abs(weights_org))

        # Assert sign(weights_new_sparse[i]) == sign(weights_org[i])
        assert_array_equal(np.sign(weights_new_sparse), np.sign(weights_org))

        # Asset weights_new_sparse == weights_new_dense
        assert_allclose(weights_new_sparse, weights_new_dense)


###############################################################################
