###############################################################################
#
# Tests for investops.check
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

from investops.check import check_corr_matrix, fix_corr_matrix
import numpy as np
from numpy.testing import assert_allclose
import pytest

###############################################################################
# check_corr_matrix()


def test_check_corr_matrix():
    """test investops.check.check_corr_matrix()."""

    # First test valid correlation matrices which should NOT raise exceptions.

    # Test 1
    corr = np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])
    check_corr_matrix(corr=corr)

    # Test 2
    corr = np.array([[1.0, 0.5, 0.3],
                     [0.5, 1.0, 0.2],
                     [0.3, 0.2, 1.0]])
    check_corr_matrix(corr=corr)

    # Test 3
    corr = np.array([[1.0, -0.5, 0.3],
                     [-0.5, 1.0, -0.2],
                     [0.3, -0.2, 1.0]])
    check_corr_matrix(corr=corr)

    # Now test invalid correlation matrices which SHOULD raise exception.

    # Test 4
    with pytest.raises(ValueError):
        corr = np.array([[1.1, 0.5, 0.3],
                         [0.5, 1.0, 0.2],
                         [0.3, 0.2, 1.0]])
        check_corr_matrix(corr=corr)

    # Test 5
    with pytest.raises(ValueError):
        corr = np.array([[1.0, -0.5, 0.3],
                         [0.5, 1.0, 0.2],
                         [0.3, 0.2, 1.0]])
        check_corr_matrix(corr=corr)

    # Test 6
    with pytest.raises(ValueError):
        corr = np.array([[1.0, 0.5, 0.3],
                         [0.5, 1.0, 1.2],
                         [0.3, 1.2, 1.0]])
        check_corr_matrix(corr=corr)

    # Test 7
    with pytest.raises(ValueError):
        corr = np.array([[1.0, 0.5, -2.3],
                         [0.5, 1.0, 0.2],
                         [-2.3, 0.2, 1.0]])
        check_corr_matrix(corr=corr)

    # Test 8
    with pytest.raises(ValueError):
        corr = np.array([[1.0, -0.5],
                         [-0.5, 1.0],
                         [0.3, -0.2]])
        check_corr_matrix(corr=corr)


###############################################################################
# fix_corr_matrix()


def test_fix_corr_matrix():
    """test investops.check.fix_corr_matrix()."""
    # Test 1
    corr = np.array([[1.0, 0.5, 0.3],
                     [0.5, 1.0, 0.2],
                     [0.3, 0.2, 1.0]])
    corr_correct = np.array([[1.0, 0.5, 0.3],
                             [0.5, 1.0, 0.2],
                             [0.3, 0.2, 1.0]])
    corr_fixed = fix_corr_matrix(corr=corr)
    assert_allclose(corr_fixed, corr_correct)

    # Test 2
    corr = np.array([[1.0, -0.5, 0.3],
                     [-0.5, 1.0, -0.2],
                     [0.3, -0.2, 1.0]])
    corr_correct = np.array([[1.0, -0.5, 0.3],
                             [-0.5, 1.0, -0.2],
                             [0.3, -0.2, 1.0]])
    corr_fixed = fix_corr_matrix(corr=corr)
    assert_allclose(corr_fixed, corr_correct)

    # Test 3
    corr = np.array([[1.0, -0.5, 0.3],
                     [-0.5, 1.0, -0.2],
                     [0.3, -0.2, 1.0]])
    corr_correct = np.array([[1.0, -0.5, 0.3],
                             [-0.5, 1.0, -0.2],
                             [0.3, -0.2, 1.0]])
    corr_fixed = fix_corr_matrix(corr=corr)
    assert_allclose(corr_fixed, corr_correct)

    # Test 4
    corr = np.array([[1.2, -0.5, 0.3],
                     [-0.5, 1.0, -0.2],
                     [0.3, -0.2, 1.0]])
    corr_correct = np.array([[1.0, -0.5, 0.3],
                             [-0.5, 1.0, -0.2],
                             [0.3, -0.2, 1.0]])
    corr_fixed = fix_corr_matrix(corr=corr)
    assert_allclose(corr_fixed, corr_correct)

    # Test 5
    corr = np.array([[1.2, 3.5, 0.3],
                     [-2.5, 1.5, -0.2],
                     [-0.3, -0.3, 0.0]])
    corr_correct = np.array([[1.0, 1.0, 0.3],
                             [1.0, 1.0, -0.2],
                             [0.3, -0.2, 1.0]])
    corr_fixed = fix_corr_matrix(corr=corr)
    assert_allclose(corr_fixed, corr_correct)


###############################################################################
