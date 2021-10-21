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
def fix_corr_matrix(corr):
    """
    Repair a correlation matrix so it is symmetrical, limited between -1 and 1,
    and the diagonal elements are all 1. The upper-triangle is copied to the
    lower-triangle. NaN values are set to zero. The data is updated inplace.

    :param corr:
        Numpy 2-dim array for the correlation matrix which is updated inplace.

    :return:
        The same Numpy array as the `corr` input argument.
    """
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
        # Check the diagonal is 1.
        if np.abs(corr[i, i] - 1.0) > tol:
            raise ValueError('Correlation matrix diagonal is not 1.')

        # For each column after the diagonal.
        for j in range(i + 1, n):
            # Check the correlations are between -1 and 1.
            if (corr[i, j] < -1.0 - tol) or (corr[i, j] > 1.0 + tol):
                msg = 'Correlation matrix has element outside range [-1,1].'
                raise ValueError(msg)

            # Check the matrix is symmetrical.
            if np.abs(corr[i, j] - corr[j, i]) > tol:
                raise ValueError('Correlation matrix is not symmetrical.')


###############################################################################