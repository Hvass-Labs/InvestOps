###############################################################################
#
# Tests for investops.normalize
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

from investops.normalize import normalize_weights, normalize_weights_pos
import numpy as np
from numpy.testing import assert_allclose

###############################################################################
# normalize_weights()

def test_normalize_weights():
    """Test investops.normalize.normalize_weights()"""
    # Test 1
    weights = np.array([0.2, -0.3, 0.4, -0.5, 0.6])
    weights_norm, cash = normalize_weights(weights=weights, limit_pos=0.8, limit_neg=-0.3, max_ratio=0.2)
    assert_allclose(weights_norm, [0.13333333, -0.06, 0.26666667, -0.1, 0.4])
    assert_allclose(cash, 0.2)

    # Test 2
    weights = np.array([0.2, -0.3, 0.4, -0.5, 0.1])
    weights_norm, cash = normalize_weights(weights=weights, limit_pos=0.8, limit_neg=-0.3, max_ratio=0.2)
    assert_allclose(weights_norm, [0.2, -0.0525, 0.4, -0.0875, 0.1])
    assert_allclose(cash, 0.3)

    # Test 3
    weights = np.array([[0.1, -0.3, 0.2, -0.5, 0.3],
                        [0.4, -0.2, 0.6, -0.4, 0.3]])
    weights_norm, cash = normalize_weights(weights=weights, limit_pos=0.8, limit_neg=-0.3, max_ratio=0.2)
    assert_allclose(weights_norm, [[0.1, -0.045, 0.2, -0.075, 0.3],
                                   [0.24615385, -0.05333333, 0.36923077, -0.10666667, 0.18461538]])
    assert_allclose(cash, [0.4, 0.2])

    # Test 4
    weights = np.array([[0.6, -0.3, 0.8, -0.5, 0.3],
                        [0.2, -0.2, 0.4, -0.4, 0.5]])
    weights_norm, cash = normalize_weights(weights=weights, limit_pos=1.2, limit_neg=-0.3, max_ratio=0.2)
    assert_allclose(weights_norm, [[0.42352941, -0.09, 0.56470588, -0.15, 0.21176471],
                                   [0.2, -0.07333333, 0.4, -0.14666667, 0.5]])
    assert_allclose(cash, [-0.2, -0.1])


###############################################################################
# normalize_weights_pos()

def test_normalize_weights_pos():
    """Test investops.normalize.normalize_weights_pos()"""
    # Test 1
    weights = np.array([0.0, 0.0, 0.0])
    weights_norm, cash = normalize_weights_pos(weights=weights, limit_pos=1.0)
    assert_allclose(weights_norm, [0.0, 0.0, 0.0])
    assert_allclose(cash, 1.0)

    # Test 2
    weights = np.array([0.1, 0.2, 0.3])
    weights_norm, cash = normalize_weights_pos(weights=weights, limit_pos=1.0)
    assert_allclose(weights_norm, [0.1, 0.2, 0.3])
    assert_allclose(cash, 0.4)

    # Test 3
    weights = np.array([0.2, 0.4, 0.6])
    weights_norm, cash = normalize_weights_pos(weights=weights, limit_pos=1.0)
    assert_allclose(weights_norm, [0.16666667, 0.33333333, 0.5])
    assert_allclose(cash, 0.0)

    # Test 4
    weights = np.array([0.2, 0.4, 0.6])
    weights_norm, cash = normalize_weights_pos(weights=weights, limit_pos=0.8)
    assert_allclose(weights_norm, [0.13333333, 0.26666667, 0.4])
    assert_allclose(cash, 0.2)

    # Test 5
    weights = np.array([[0.2, 0.4, 0.6],
                        [0.4, 0.6, 0.8]])
    weights_norm, cash = normalize_weights_pos(weights=weights, limit_pos=0.8)
    assert_allclose(weights_norm, [[0.13333333, 0.26666667, 0.4],
                                   [0.17777778, 0.26666667, 0.35555556]])
    assert_allclose(cash, [0.2, 0.2])

    # Test 6
    weights = np.array([[0.2, 0.4, 0.6],
                        [0.4, 0.6, 0.8]])
    weights_norm, cash = normalize_weights_pos(weights=weights, limit_pos=1.4)
    assert_allclose(weights_norm, [[0.2, 0.4, 0.6],
                                   [0.31111111, 0.46666667, 0.62222222]])
    assert_allclose(cash, [-0.2, -0.4])


###############################################################################
