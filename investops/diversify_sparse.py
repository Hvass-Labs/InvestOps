###############################################################################
#
# Functions for diversifying an investment portfolio when the
# correlation matrix is sparse so it mostly contains zeros.
#
# This implements the sparse version of "Hvass Diversification" from the paper:
# - M.E.H. Pedersen, "Fast Portfolio Diversification", 2022.
#   https://ssrn.com/abstract=4009041
#   https://github.com/Hvass-Labs/Finance-Papers
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

import numpy as np
import pandas as pd
from numba import jit
from investops.diversify import _check_convergence
from investops.sparse import sparse_corr_to_numpy
from investops.check import check_weights

###############################################################################
# Helper functions.

def _fix_input(weights_org, corr_coef, weights_guess=None):
    """
    Ensure the weights and correlation coefficients do not have NaN-values
    (Not-a-Number) by filling with 0.

    This returns a copy of the input data.

    :param weights_org:
        Numpy array with the originally desired portfolio weights.

    :param corr_coef:
        Numpy array with the correlation coefficients of a sparse matrix.

    :param weights_guess:
        Numpy array with portfolio weights for the starting guess.

    :return:
        - New Numpy array with portfolio weights.
        - New Numpy array with the correlation coefficients of a sparse matrix.
        - New Numpy array with the initial portfolio weights guess.
    """
    # Copy arrays and fill NaN-values with 0.0
    weights_org = np.nan_to_num(weights_org, nan=0.0, copy=True)
    corr_coef = np.nan_to_num(corr_coef, nan=0.0, copy=True)
    if weights_guess is not None:
        weights_guess = np.nan_to_num(weights_guess, nan=0.0, copy=True)

    return weights_org, corr_coef, weights_guess


def _check_pandas_index(weights_org, weights_guess=None):
    """
    If the arguments are Pandas Series or DataFrames, then check that their
    indices have matching names, otherwise throw a `ValueError`.

    This is because Pandas can automatically align the data when doing math
    operations on the data, but we will be using Numpy in these algorithms,
    so the data would get corrupted if the index names don't match.

    :param weights_org:
        Array with the original asset-weights.

    :param weights_guess:
        Array with a starting guess for the diversified asset-weights.

    :raises:
        `ValueError` if the args have mis-matched Pandas index.

    :return:
        None
    """
    # Booleans whether the args are Pandas data-types.
    is_pandas_org = isinstance(weights_org, (pd.Series, pd.DataFrame))
    is_pandas_guess = isinstance(weights_guess, (pd.Series, pd.DataFrame))

    # Check weights_org and weights_guess.
    if is_pandas_org and is_pandas_guess:
        if not weights_org.index.equals(weights_guess.index):
            msg = 'Mismatch in index names for Pandas data.'
            raise ValueError(msg)

###############################################################################
# Full Exposure.

@jit(parallel=False)
def full_exposure_sparse(weights, corr_i, corr_j, corr_coef):
    """
    Calculate the so-called "Full Exposure" for each asset, which measures
    the entire portfolio's exposure to each asset both directly through the
    asset weights themselves, but also indirectly through their correlations
    with other assets in the portfolio.

    This function is decorated with Numba Jit, which means it compiles into
    super-fast machine-code the first time it is run. Note that this function
    currently does NOT support parallel execution.

    The sparse correlation matrix is given in Coordinate (COO) format
    with 3 arrays for the matrix coordinates and correlation coefficients.

    NOTE! The arguments must be either Numpy arrays or Python lists.
    The arguments cannot be Pandas Series or DataFrames, because Numba Jit
    does not support Pandas. The optimal run-speed is when using Numpy arrays.

    WARNING! It is very important that the sparse matrix ONLY contains the
    upper-triangle matrix, and it must NOT contain the diagonal and
    lower-triangle matrix, because the algorithm would then double-count
    the correlations and make faulty the calculations.

    The sparse algorithm for the Full Exposure is from the following paper:
    - M.E.H. Pedersen, "Fast Portfolio Diversification", 2022.
      https://ssrn.com/abstract=4009041
      https://github.com/Hvass-Labs/Finance-Papers

    :param weights:
        Numpy array with floats for the portfolio weights for the assets.

    :param corr_i:
        Numpy array with ints for the row indices of the sparse corr matrix.

    :param corr_j:
        Numpy array with ints for the column indices of the sparse corr matrix.

    :param corr_coef:
        Numpy array with floats for the correlation coefficients of the
        sparse correlation matrix.

    :returns:
        Numpy array with the Full Exposure of each asset.
    """
    # Initialize the array of correlated exposures with the squared weights,
    # because this is the minimum possible value for the sum of corr. exposures
    # if all the asset's correlation coefficients are zero.
    sum_corr_exp = np.copy(weights) ** 2

    # For each correlation coefficient in the sparse matrix.
    # WARNING! There is a "race condition" when this loop is run in parallel,
    # because the array sum_corr_exp is read and written by all the threads.
    for i, j, c in zip(corr_i, corr_j, corr_coef):
        # The indices i and j are for the row and column of the corr. matrix,
        # and c is the correlation coefficient at that position in the matrix.

        # Portfolio weights of assets i and j.
        w_i = weights[i]
        w_j = weights[j]

        # Product of the two asset weights and their correlation.
        prod = w_i * w_j * c

        # If the product is positive then the correlation is deemed "bad"
        # and must be included in the calculation of the Full Exposure,
        # so the two portfolio weights can be adjusted accordingly.
        # This is explained in Section 2.1 of the paper referenced above.
        if prod > 0.0:
            # Multiply with the correlation again, because otherwise the
            # square-root calculated below would amplify the correlation.
            # Because this can result in a negative number, we also need
            # to take the absolute value.
            prod_final = np.abs(prod * c)

            # Update the sum of correlation exposures for assets i and j.
            # Because the correlation is symmetrical, the two assets are
            # updated identically.
            sum_corr_exp[i] += prod_final
            sum_corr_exp[j] += prod_final

    # Finalize the calculation of the Full Exposure.
    full_exp = np.sign(weights) * np.sqrt(sum_corr_exp)

    return full_exp

###############################################################################
# Diversification algorithm.

@jit(parallel=False)
def _update_weights_sparse(weights_org, weights_new,
                           corr_i, corr_j, corr_coef):
    """
    Helper-function for the function `diversify_weights_sparse` which performs
    a single update of the portfolio weights, using the algorithm from
    Section 3.1 of the paper referenced above.

    The sparse correlation matrix is given in Coordinate (COO) format
    with 3 arrays for the matrix coordinates and correlation coefficients.

    NOTE! The arguments should be Numpy arrays for optimal speed.

    WARNING! It is very important that the sparse matrix ONLY contains the
    upper-triangle matrix, and it must NOT contain the diagonal and
    lower-triangle matrix, because the algorithm would then double-count
    the correlations and make faulty calculations.

    :param weights_org:
        Numpy array with floats for the original portfolio weights.

    :param weights_new:
        Numpy array with the adjusted portfolio weights. It is updated inplace.

    :param corr_i:
        Numpy array with ints for the row indices of the sparse corr matrix.

    :param corr_j:
        Numpy array with ints for the column indices of the sparse corr matrix.

    :param corr_coef:
        Numpy array with floats for the correlation coefficients of the
        sparse correlation matrix.

    :return:
        Float with the max absolute difference between the Full Exposure
        and the original portfolio weights. This is used to abort the outer
        algorithm's for-loop when sufficiently good weights have been found.
    """
    # This could be implemented entirely with Numpy, but would need special
    # handling in case the Full Exposure is zero for some assets. This is a
    # slightly cleaner implementation that is also faster with Numba jit.

    # Calculate the Full Exposure for all the portfolio weights.
    full_exp = full_exposure_sparse(weights=weights_new,
                                    corr_i=corr_i, corr_j=corr_j,
                                    corr_coef=corr_coef)

    # Init. max abs difference between the Full Exposure and original weights.
    max_abs_dif = 0.0

    # Number of portfolio weights.
    n = len(weights_org)

    # For each portfolio weight.
    # WARNING! There is a "race condition" when this loop is run in parallel,
    # because the variable max_abs_dif is read and written by all the threads.
    for i in range(n):
        # Original weight for asset i.
        w_org_i = weights_org[i]

        # Full Exposure for asset i.
        full_exp_i = full_exp[i]

        # If the Full Exposure is non-zero.
        if full_exp_i != 0.0:
            # Update the portfolio weight for asset i.
            weights_new[i] *= w_org_i / full_exp_i

        # Update max abs difference between Full Exposure and original weight.
        abs_dif = np.abs(full_exp_i - w_org_i)
        if abs_dif > max_abs_dif:
            max_abs_dif = abs_dif

    return max_abs_dif


def diversify_weights_sparse(weights_org, corr_i, corr_j, corr_coef,
                             weights_guess=None, fix_input=True,
                             log=None, max_iter=100, tol=1e-3):
    """
    Find new asset-weights that minimize the Mean Squared Error (MSE) between
    the original asset-weights and the Full Exposure of the new asset-weights.

    This is the sparse version of the algorithm, which is generally only faster
    than the dense version of the algorithm, if the correlation matrix is very
    sparse so most of the correlation coefficients are zero.
    See the paper referenced below for timing experiments.

    The sparse correlation matrix is given in Coordinate (COO) format
    with 3 arrays for the matrix coordinates and correlation coefficients.

    WARNING! It is very important that the sparse matrix ONLY contains the
    upper-triangle matrix, and it must NOT contain the diagonal and
    lower-triangle matrix, because the algorithm would then double-count
    the correlations and make faulty calculations.

    The sparse version of the diversification algorithm is from this paper:
    - M.E.H. Pedersen, "Fast Portfolio Diversification", 2022.
      https://ssrn.com/abstract=4009041
      https://github.com/Hvass-Labs/Finance-Papers

    :param weights_org:
        Array with the originally desired asset-weights for the portfolio.
        These can be either positive or negative, and they need not sum to 1.
        This data can either be a Pandas Series or Numpy array.

    :param corr_i:
        Array or list of the row indices of the sparse correlation matrix.
        If `weights_org` is a Numpy array, then this is a Numpy array of ints.
        If `weights_org` is a Pandas Series, then this is a list of strings.

    :param corr_j:
        Array or list of the column indices of the sparse correlation matrix.
        If `weights_org` is a Numpy array, then this is a Numpy array of ints.
        If `weights_org` is a Pandas Series, then this is a list of strings.

    :param corr_coef:
        Numpy array with floats for the correlation coefficients of the
        sparse correlation matrix.

    :param weights_guess:
        Array with a starting guess for the adjusted portfolio weights.
        If you are calling this function with `weights_org` and `corr` being
        nearly identical on each call, then you might save computation time
        by passing the last weights that were output from this function as
        the arg `weights_guess` the next time you call this function. This
        may reduce the number of iterations needed for convergence.

    :param fix_input:
        Boolean whether to repair input by filling NaN-values (Not-a-Number)
        in `weights_org`, `weights_guess` and `corr_coef`.

    :param log:
        If this is a list-like object then it will have its function `append`
        called after each iteration with the new weights, so you can print
        them later. This is useful for debugging and other demonstrations.

    :param max_iter:
        Max iterations of the algorithm.

    :param tol:
        Stop the algorithm when asset-weight adjustments are smaller than this
        tolerance level.

    :return:
        Array with the adjusted asset-weights.
        If `weights_org` is a Pandas Series or DataFrame,
        then this is also a Pandas Series, otherwise this is a Numpy array.
    """
    # If using Pandas data-types, ensure their index names match.
    # Note: This does NOT check the sparse correlation matrix,
    # which is tested during the conversion process below.
    _check_pandas_index(weights_org=weights_org, weights_guess=weights_guess)

    # Convert weights_org and sparse corr. matrix from Pandas to Numpy.
    if isinstance(weights_org, (pd.Series, pd.DataFrame)):
        # Save the Pandas index for later use with the return-data.
        index = weights_org.index

        # Convert the weights and sparse correlation matrix to Numpy arrays.
        # Args corr_i and corr_j are assumed to be lists of strings which must
        # be present in the index of weights_org, otherwise an error is raised.
        weights_org, corr_i, corr_j, corr_coef = \
            sparse_corr_to_numpy(weights=weights_org, corr_i=corr_i,
                                 corr_j=corr_j, corr_coef=corr_coef)
    else:
        # This is also used to indicate that the input was NOT Pandas data.
        index = None

        # Ensure the sparse correlation arrays are in Numpy format.
        # This may or may not be a copy of the data.
        corr_i = np.asarray(corr_i)
        corr_j = np.asarray(corr_j)
        corr_coef = np.asarray(corr_coef)

    # Convert weights_guess from Pandas to Numpy.
    if isinstance(weights_guess, (pd.Series, pd.DataFrame)):
        # This may or may not be a copy of the data.
        # Note: Flatten is necessary if it is a Pandas DataFrame.
        weights_guess = weights_guess.to_numpy().flatten()

    # Ensure the weights and correlation coefficients are valid.
    if fix_input:
        # This copies the data.
        weights_org, corr_coef, weights_guess = \
            _fix_input(weights_org=weights_org, corr_coef=corr_coef,
                       weights_guess=weights_guess)

    # Select a starting point for the new adjusted weights.
    # The arrays are copied so we don't modify the argument data.
    # It is possible that the data was already copied above, so there
    # is a slight redundancy here, but it makes the code easier to read.
    if weights_guess is not None:
        # Use the guessed weights as the starting point.
        # In case a guessed weight is zero, use the original weight,
        # otherwise the weight-adjustment would always get stuck in zero.
        # This should create a new numpy array so there is no need to copy.
        weights_new = np.where(weights_guess != 0.0, weights_guess, weights_org)
    else:
        # Use a copy of the original weights as the starting point.
        weights_new = np.copy(weights_org)

    # Log the initial weights?
    if log is not None:
        # Array is copied because the update iterates on the same array, so
        # the entire log would be filled with the same values if not copied.
        log.append(weights_new.copy())

    # Repeat for a number of iterations or until convergence
    # which breaks out of the for-loop further below.
    for i in range(max_iter):
        # Update the array weights_new inplace.
        max_abs_dif = _update_weights_sparse(weights_org=weights_org,
                                             weights_new=weights_new,
                                             corr_i=corr_i, corr_j=corr_j,
                                             corr_coef=corr_coef)

        # Log the updated weights?
        if log is not None:
            # Array is copied because the update iterates on the same array, so
            # the entire log would be filled with the same values if not copied.
            log.append(weights_new.copy())

        # Abort the for-loop when converged to a solution.
        if max_abs_dif < tol:
            break

    # Check that we have converged to a sufficiently good solution.
    _check_convergence(max_abs_dif=max_abs_dif, tol=tol)

    # Check that the original and new portfolio weights are consistent.
    check_weights(weights_org=weights_org, weights_new=weights_new)

    # If the input weights_org was Pandas data, then also output Pandas data.
    if index is not None:
        weights_new = pd.Series(data=weights_new, index=index)

    return weights_new

###############################################################################
# Convert log.

def log_to_dataframe_sparse(weights_org, corr_i, corr_j, corr_coef, log):
    """
    Convert the log from `diversify_weights_sparse` to a Pandas DataFrame which
    shows the iterations of the adj. portfolio weights and their Full Exposure.

    The sparse correlation matrix is given in Coordinate (COO) format
    with 3 arrays for the matrix coordinates and correlation coefficients.

    WARNING! It is very important that the sparse matrix ONLY contains the
    upper-triangle matrix, and it must NOT contain the diagonal and
    lower-triangle matrix, because the algorithm would then double-count
    the correlations and make faulty calculations.

    :param weights_org:
        Numpy array with floats for the originally desired portfolio weights.

    :param corr_i:
        Numpy array with ints for the row indices of the sparse corr matrix.

    :param corr_j:
        Numpy array with ints for the column indices of the sparse corr matrix.

    :param corr_coef:
        Numpy array with floats for the correlation coefficients of the
        sparse correlation matrix.

    :param log:
        List of Numpy arrays with portfolio weights. This is obtained by first
        passing the list as the `log` arg in the `diversify_weights_sparse`
        function.

    :return:
        Pandas DataFrame
    """
    # Convert log to numpy 2-dim array.
    log_weights = np.array(log)

    # Get the number of iterations and assets in the log.
    num_iterations, num_assets = log_weights.shape

    # Initialize log for the Full Exposure.
    log_full_exp = []

    # Initialize log for the Mean Squared Error (MSE).
    log_mse = []

    # For each array of adjusted weights in the log.
    for weights_new in log_weights:
        # Calculate the Full Exposure of the logged weights.
        fe = full_exposure_sparse(weights=weights_new, corr_i=corr_i,
                                  corr_j=corr_j, corr_coef=corr_coef)

        # Save the results.
        log_full_exp.append(fe)

        # Calculate the Mean Squared Error (MSE).
        mse = np.mean((fe - weights_org) ** 2)

        # Save the results.
        log_mse.append(mse)

    # Combine the arrays of adjusted weights and Full Exposure, so that:
    # 1st column is for 1st weights, 2nd column is for 1st Full Exposure.
    # 3rd column is for 2nd weights, 4th column is for 2nd Full Exposure.
    data = np.dstack((log_weights, log_full_exp)).reshape(num_iterations, -1)

    # Generate names for the columns.
    names = []
    for i in range(1, num_assets + 1):
        names.append(f'Weight {i}')
        names.append(f'Full Exp. {i}')

    # Index for the rows.
    index = pd.Series(data=list(range(0, num_iterations)), name='Iteration')

    # Create Pandas DataFrame with the data.
    df = pd.DataFrame(data=data, columns=names, index=index)

    # Append a column for the Mean Squared Error (MSE).
    df['MSE'] = log_mse

    return df

###############################################################################
