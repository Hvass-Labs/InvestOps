###############################################################################
#
# Tests for investops.sparse
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

from investops.sparse import (sparse_corr_to_numpy,
                              matrix_to_sparse, sparse_to_matrix)
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

###############################################################################
# matrix_to_sparse

def test_matrix_to_sparse():
    """Test investops.sparse.matrix_to_sparse()"""
    # Test 1
    mat = np.array([[1.0, 0.5, -0.2],
                    [0.0, 0.0, 0.3],
                    [-0.1, 0.0, 1.0]])
    sparse_i, sparse_j, sparse_v = matrix_to_sparse(mat=mat)
    assert_array_equal(sparse_i, [0, 0, 0, 1, 2, 2])
    assert_array_equal(sparse_j, [0, 1, 2, 2, 0, 2])
    assert_allclose(sparse_v, [1.0, 0.5, -0.2, 0.3, -0.1, 1.0])

    # Test 2
    mat = np.array([[1.0, 0.5, -0.2],
                    [0.1, 0.2, 0.3],
                    [-0.1, 0.5, -1.0]])
    sparse_i, sparse_j, sparse_v = matrix_to_sparse(mat=mat)
    assert_array_equal(sparse_i, [0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert_array_equal(sparse_j, [0, 1, 2, 0, 1, 2, 0, 1, 2])
    assert_allclose(sparse_v, [1.0, 0.5, -0.2, 0.1, 0.2, 0.3, -0.1, 0.5, -1.0])


###############################################################################
# sparse_to_matrix

def test_sparse_to_matrix():
    """Test investops.sparse.sparse_to_matrix()"""
    # Test 1
    sparse_i = np.array([0, 1, 2])
    sparse_j = np.array([2, 1, 1])
    sparse_v = np.array([-1.5, 0.5, 2.2])
    mat = sparse_to_matrix(sparse_i=sparse_i, sparse_j=sparse_j,
                           sparse_v=sparse_v, shape=(3, 3))
    assert_array_equal(mat, [[0.0, 0.0, -1.5],
                             [0.0, 0.5, 0.0],
                             [0.0, 2.2, 0.0]])

    # Test 2
    sparse_i = np.array([0, 1, 2])
    sparse_j = np.array([2, 1, 1])
    sparse_v = np.array([-1.5, np.nan, 2.2])
    mat = sparse_to_matrix(sparse_i=sparse_i, sparse_j=sparse_j,
                           sparse_v=sparse_v, shape=(3, 3))
    assert_array_equal(mat, [[0.0, 0.0, -1.5],
                             [0.0, np.nan, 0.0],
                             [0.0, 2.2, 0.0]])

###############################################################################
# sparse_corr_to_numpy

def test_sparse_corr_to_numpy():
    """Test investops.sparse.sparse_corr_to_numpy()"""
    # Test 1
    weights = pd.Series(dict(MSFT=0.1, BBBY=0.3, AAPL=0.2))
    corr_sparse = [('MSFT', 'BBBY', 0.5), ('AAPL', 'BBBY', 0.1)]
    corr_i, corr_j, corr_coef = zip(*corr_sparse)
    weights_np, corr_i_np, corr_j_np, corr_coef_np = \
        sparse_corr_to_numpy(weights=weights, corr_i=corr_i,
                             corr_j=corr_j, corr_coef=corr_coef)
    assert_allclose(weights_np, [0.1, 0.3, 0.2])
    assert_array_equal(corr_i_np, [0, 2])
    assert_array_equal(corr_j_np, [1, 1])
    assert_allclose(corr_coef_np, [0.5, 0.1])

###############################################################################
