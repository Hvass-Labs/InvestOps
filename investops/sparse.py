###############################################################################
#
# Functions for sparse matrices.
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

import pandas as pd
import numpy as np
from numba import jit

###############################################################################
# Convert sparse correlation matrix.

def sparse_corr_to_numpy(weights, corr_i, corr_j, corr_coef):
    """
    Convert portfolio weights and a sparse correlation matrix to Numpy arrays.
    This is ONLY to be used when `weights` is a Pandas DataFrame or Series,
    which uses strings instead of numbers for indexing assets in the portfolio,
    e.g. when using stock-tickers such as 'MSFT' or 'AMZN'.

    The sparse matrix is in Coordinate (COO) format, where the rows are given
    in `corr_i`, and the columns are given in `corr_j`, both of which must be
    lists of strings that are found in the index of `weights`. The correlation
    coefficients for those positions in the matrix are given in `corr_coef`.

    The outputs are Numpy arrays where all the strings that identify the assets
    in the portfolio have been converted to integers that index the weights.

    :param weights:
        Pandas DataFrame or Series that maps strings such as stock-tickers
        to floating-point values for the portfolio weights.

    :param corr_i:
        Iterator with row-indices for the correlation matrix. These must
        be some of the same strings used in the index of `weights`.

    :param corr_j:
        Iterator with column-indices for the correlation matrix. These must
        be some of the same strings used in the index of `weights`.

    :param corr_coef:
        Iterator with the correlation coefficients at position (corr_i, corr_j)
        of the correlation matrix.

    :raises:
        - TypeError if `weights` is not a Pandas DataFrame or Series.
        - KeyError if a string in `corr_i` or `corr_j` is not found in
          the index of `weights`.

    :return:
        - weights_np: Numpy array with floats for the portfolio weights.
        - corr_i_np: Numpy array with ints for row-indices of corr. matrix.
        - corr_j_np: Numpy array with ints for col-indices of corr. matrix.
        - corr_coef_np: Numpy array with floats for corr. coefficients.
    """
    # Check the weights are a Pandas data-types.
    if not isinstance(weights, (pd.DataFrame, pd.Series)):
        msg = 'Argument \'weights\' must be a Pandas DataFrame or Series.'
        raise TypeError(msg)

    # Get the list of strings e.g. stock-tickers used as index in the weights.
    index = weights.index

    # Create a map from the strings used in the index to their positions.
    # This allows fast conversion of the sparse correlation matrix below.
    index_map = dict(zip(index, range(len(index))))

    # Convert weights from Pandas to Numpy. This may be a copy of the data.
    # Note: Flatten is necessary if it is a Pandas DataFrame.
    weights_np = weights.to_numpy().flatten()

    # Convert coordinates of the sparse correlation matrix to integers.
    # This raises an exception if a value is not in the index of the weights.
    corr_i_list = [index_map[i] for i in corr_i]
    corr_j_list = [index_map[j] for j in corr_j]

    # Convert to Numpy arrays.
    corr_i_np = np.asarray(corr_i_list)
    corr_j_np = np.asarray(corr_j_list)
    corr_coef_np = np.asarray(corr_coef)

    return weights_np, corr_i_np, corr_j_np, corr_coef_np

###############################################################################
# Convert between dense and sparse matrix formats.

@jit
def matrix_to_sparse(mat):
    """
    Convert a 2-dim Numpy array to a sparse matrix in Coordinate (COO) format,
    which is given by 3 arrays for the row and column indices, and the values.

    :param mat:
        Numpy 2-dim array / matrix.
    
    :return:
        - Numpy array of the row-indices i for the sparse matrix.
        - Numpy array of the column-indices j for the sparse matrix.
        - Numpy array of the non-zero values for the sparse matrix. 
    """
    # Initialize empty lists for the sparse matrix coordinates and values.
    sparse_i = []
    sparse_j = []
    sparse_v = []

    # For each row in the matrix.
    for i in range(mat.shape[0]):
        # For each column in the matrix.
        for j in range(mat.shape[1]):
            # Matrix value at this position.
            v = mat[i, j]

            # If value is non-zero then append to sparse matrix.
            if v != 0:
                sparse_i.append(i)
                sparse_j.append(j)
                sparse_v.append(v)

    # Convert to Numpy arrays.
    sparse_i = np.array(sparse_i)
    sparse_j = np.array(sparse_j)
    sparse_v = np.array(sparse_v)

    return sparse_i, sparse_j, sparse_v


@jit
def sparse_to_matrix(sparse_i, sparse_j, sparse_v, shape, dtype=np.float64):
    """
    Convert a sparse matrix in Coordinate (COO) format to a 2-dim Numpy array.

    :param sparse_i:
        Numpy array with integers for the row-indices of the sparse matrix.

    :param sparse_j:
        Numpy array with integers for the column-indices of the sparse matrix.

    :param sparse_v:
        Numpy array with floats for the values of the sparse matrix.

    :param shape:
        Tuple with the number of rows and columns for the output Numpy array.

    :param dtype:
        Data-type for the output Numpy array.

    :return:
        Numpy 2-dim array.
    """
    # Initialize the new array with zeros.
    mat = np.zeros(shape=shape, dtype=dtype)

    # Populate the Numpy array with the values from the sparse matrix.
    for i, j, v in zip(sparse_i, sparse_j, sparse_v):
        mat[i, j] = v

    return mat

###############################################################################
