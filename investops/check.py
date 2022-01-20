###############################################################################
#
# Functions for checking and repairing data.
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
from numba import jit

###############################################################################


@jit
def fix_corr_matrix(corr, copy=True):
    """
    Repair a correlation matrix so it is symmetrical, limited between -1 and 1,
    and the diagonal elements are all 1. The upper-triangle is copied to the
    lower-triangle. NaN values are set to zero.

    :param corr:
        Numpy 2-dim array for the correlation matrix which is updated inplace.

    :param copy:
        Boolean whether to copy `corr` (True) or update `corr` inplace (False).

    :return:
        Numpy array with the repaired correlation matrix.
    """
    # Update the data inplace or make a copy?
    if copy:
        corr = corr.copy()

    # Number of rows and columns.
    n = len(corr)

    # For each row.
    for i in range(n):
        # For each column after the diagonal.
        for j in range(i + 1, n):
            # Get the correlation value.
            c = corr[i, j]

            #  Ensure the correlation value is valid.
            if np.isnan(c):
                # NaN (Not-a-Number) value is set to zero.
                c = 0.0
            elif c > 1.0:
                # Clip the value if it is higher than 1.0
                c = 1.0
            elif c < -1.0:
                # Clip the value if it is lower than -1.0
                c = -1.0

            # Update the matrix inplace. Also copy to the lower-triangle.
            corr[i, j] = corr[j, i] = c

        # Ensure the diagonal is 1.
        corr[i, i] = 1.0

    return corr


@jit
def check_corr_matrix(corr, tol=1e-9):
    """
    Check that a numpy array is a valid correlation matrix:

    - It must be matrix-shaped.
    - Its elements must be between -1 and 1.
    - The diagonal must be 1.
    - The matrix must be symmetrical.

    The checks allow for small floating point rounding errors.

    :param corr:
        Numpy 2-dim array for the correlation matrix.
        Note: It is NOT checked that it is a valid Numpy array, because that
        kind of type-checking is not supported inside a Numba Jit function.

    :param tol:
        Float with the error tolerance in the float comparisons.

    :raises:
        `ValueError` if the `corr` arg is an invalid correlation matrix.

    :return:
        None
    """
    # Assume `corr` is a valid Numpy array, because we cannot check its type
    # inside a Numba Jit function using e.g. isinstance(corr, np.ndarray).

    # Check it is matrix-shaped.
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError('Correlation matrix is not matrix-shaped.')

    # Number of rows and columns.
    n = corr.shape[0]

    # For each row in the correlation matrix.
    for i in range(n):
        # Get diagonal of the matrix.
        corr_ii = corr[i, i]

        # Check the diagonal is not NaN (Not-a-Number) or Infinity.
        if np.isnan(corr_ii) or np.isinf(corr_ii):
            raise ValueError('Correlation matrix has NaN or Infinity.')

        # Check the diagonal is 1.
        if np.abs(corr_ii - 1.0) > tol:
            raise ValueError('Correlation matrix diagonal is not 1.')

        # For each column after the diagonal.
        for j in range(i + 1, n):
            # Get elements from the correlation matrix.
            corr_ij = corr[i, j]
            corr_ji = corr[j, i]

            # Check the correlations are not NaN (Not-a-Number) or Infinity.
            if np.isnan(corr_ij) or np.isinf(corr_ij):
                raise ValueError('Correlation matrix has NaN or Infinity.')

            # Check the correlations are between -1 and 1.
            if (corr_ij < -1.0 - tol) or (corr_ij > 1.0 + tol):
                msg = 'Correlation matrix has element outside range [-1,1].'
                raise ValueError(msg)

            # Check the matrix is symmetrical.
            if np.abs(corr_ij - corr_ji) > tol:
                raise ValueError('Correlation matrix is not symmetrical.')


###############################################################################
# Check portfolio weights.

@jit
def _check_weights(weights_org, weights_new):
    """
    Helper-function for the `check_weights` function which returns the index
    of the first problem for the portfolio weights. Runs fast with Numba Jit.

    :param weights_new:
        Numpy array with the new asset-weights.

    :param weights_org:
        Numpy array with the original asset-weights.

    :return:
        `None` if no problems were found.
         Otherwise an integer with the index of the first problem.
    """
    # Number of weights.
    n = len(weights_new)

    # For each weight index.
    for i in range(n):
        # Get the weights.
        w_new = weights_new[i]
        w_org = weights_org[i]

        # Check if there is a problem and then return the corresponding index.
        # We must ensure the weight signs are equal and magnitudes are valid.
        # But because np.sign(0.0)==0.0 the check for signs is a bit awkward.
        if (np.sign(w_new) != 0.0 and np.sign(w_new) != np.sign(w_org)) or \
                (np.abs(w_new) > np.abs(w_org)):
            return i

    # No problems were found.
    return None


def check_weights(weights_org, weights_new):
    """
    Check that the original and new portfolio weights are consistent.
    They must have the same sign, and the absolute values of the new weights
    must be smaller than the absolute values of the original weights:

    (1)     sign(weights_new[i]) == sign(weights_org[i])
    (2)     abs(weights_new[i]) <= abs(weights_org[i])

    This function only takes 3.5 micro-seconds to run for 1000 weights using a
    Numba Jit implementation. A Numpy implementation would be much slower. But
    it must be split into two functions, because Numba Jit does not properly
    support the string operations used to generate the exception.

    :param weights_new:
        Numpy array with the new asset-weights.

    :param weights_org:
        Numpy array with the original asset-weights.

    :raises:
        `RuntimeError` if the weights are inconsistent.

    :return:
        None
    """
    # Get index of the first problem / inconsistency of the weights.
    idx = _check_weights(weights_org=weights_org, weights_new=weights_new)

    # If a problem was found then raise an exception.
    if idx is not None:
        msg = f'Checking the weights failed at: i={idx}, ' + \
              f'weights_new[i]={weights_new[idx]:.2e}, ' + \
              f'weights_org[i]={weights_org[idx]:.2e}'
        raise RuntimeError(msg)


###############################################################################
