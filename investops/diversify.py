###############################################################################
#
# Functions for diversifying an investment portfolio.
#
# This implements so-called "Hvass Diversification" from the paper:
# - M.E.H. Pedersen, "Simple Portfolio Optimization That Works!", 2021.
#   https://ssrn.com/abstract=3942552
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
# Copyright 2021 by Magnus Erik Hvass Pedersen
#
###############################################################################

import numpy as np
import pandas as pd
from numba import jit, prange
from investops.check import fix_corr_matrix, check_weights

###############################################################################
# Helper functions.

def _fix_input(weights_org, corr, weights_guess=None):
    """
    Ensure the weights do not have NaN-values (Not-a-Number) by filling with 0.
    And make various repairs to the correlation-matrix so it is valid.

    This makes and returns a copy of the input data.

    :param weights_org:
        Numpy array with the originally desired portfolio weights.

    :param corr:
        Numpy 2-dim array with the correlation-matrix.

    :param weights_guess:
        Numpy array with portfolio weights for the starting guess.

    :return:
        New Numpy array with portfolio weights.
        New Numpy 2-dim array with the correlation matrix.
        New Numpy array with the initial portfolio weights guess.
    """
    # Copy arrays and fill NaN-values with 0.0
    weights_org = np.nan_to_num(weights_org, nan=0.0, copy=True)
    if weights_guess is not None:
        weights_guess = np.nan_to_num(weights_guess, nan=0.0, copy=True)

    # Repair the correlation-matrix. This makes a copy of the data.
    corr = fix_corr_matrix(corr=corr, copy=True)

    return weights_org, corr, weights_guess


def _check_pandas_index(weights_org, corr, weights_guess=None):
    """
    If the arguments are Pandas Series or DataFrames, then check that their
    indices and columns have matching names, otherwise throw a `ValueError`.

    This is because Pandas can automatically align the data when doing math
    operations on the data, but we will be using Numpy in these algorithms,
    so the data would get corrupted if index and column names don't match.

    The time-usage is around 0.1 milli-seconds when `len(weights_org) == 1000`.

    :param weights_org:
        Array with the original asset-weights.

    :param corr:
        Matrix with the correlations between assets.

    :param weights_guess:
        Array with a starting guess for the diversified asset-weights.

    :raises:
        `ValueError` if the args have mis-matched Pandas index and column.

    :return:
        None
    """
    # Booleans whether the args are Pandas data-types.
    is_pandas_org = isinstance(weights_org, (pd.Series, pd.DataFrame))
    is_pandas_corr = isinstance(corr, pd.DataFrame)
    is_pandas_guess = isinstance(weights_guess, (pd.Series, pd.DataFrame))

    # Error message.
    msg = 'Mismatch in index / column names for Pandas data.'

    # Check weights_org and corr.
    if is_pandas_org and is_pandas_corr:
        if not (weights_org.index.equals(corr.index) and
                weights_org.index.equals(corr.columns)):
            raise ValueError(msg)

    # Check weights_org and weights_guess.
    if is_pandas_org and is_pandas_guess:
        if not weights_org.index.equals(weights_guess.index):
            raise ValueError(msg)

    # Check weights_guess and corr.
    # This is only necessary if weights_org is not a Pandas data-type,
    # otherwise we would already know that weights_org matches corr and
    # weights_org matches weights_guess, therefore weights_guess matches corr.
    if (not is_pandas_org) and is_pandas_guess and is_pandas_corr:
        if not (weights_guess.index.equals(corr.index) and
                weights_guess.index.equals(corr.columns)):
            raise ValueError(msg)


def _check_convergence(max_abs_dif, tol):
    """
    Check the adjusted portfolio weights have converged, so the Full Exposure
    of the portfolio weights are sufficiently close to the original weights.

    :param max_abs_dif:
        Float with max absolute difference between the Full Exposure and the
        original portfolio weights.

    :param tol:
        Tolerance level for the max abs difference.

    :raises:
        `RuntimeError` if the weights did not converge to the Full Exposure.

    :return:
        None
    """
    if max_abs_dif > tol:
        msg = 'Weights did not converge: ' + \
              f'max_abs_dif={max_abs_dif:.2e}, tol={tol:.2e}'
        raise RuntimeError(msg)


###############################################################################
# Full Exposure.

@jit(parallel=False)
def full_exposure(weights, corr):
    """
    Calculate the so-called "Full Exposure" for each asset, which measures
    the entire portfolio's exposure to each asset both directly through the
    asset weights themselves, but also indirectly through their correlations
    with other assets in the portfolio.

    There are different ways of defining the Full Exposure, as explained in
    the paper referenced below. This particular formula is Eq.(38) in that
    paper, which was found to work well in practice.

    The function is decorated with Numba Jit, which means it compiles into
    super-fast machine-code the first time it is run. This function is the most
    expensive part of the diversification method because it has time-complexity
    O(n^2) where n is the number of assets in the portfolio. Implementing it
    with for-loops instead of Numpy arrays, means that it avoids new memory
    allocations for large n^2 matrices, so the machine-code is very fast.

    The parallel version of this function is named `full_exposure_par` which
    can run even faster for large portfolios of e.g. 1000 assets or more.
    But for smaller portfolios of only e.g. 100 assets, the parallel overhead
    makes it run a bit slower, so you should test which is fastest for you.

    Note that the arguments must be Python lists or Numpy arrays and cannot be
    Pandas Series and DataFrames, because Numba Jit does not support Pandas.

    The Full Exposure is explained in more detail in the following paper:
    - M.E.H. Pedersen, "Simple Portfolio Optimization That Works!", 2021.
      https://ssrn.com/abstract=3942552
      https://github.com/Hvass-Labs/Finance-Papers

    :param weights:
        Numpy array with the portfolio weights for the assets.

    :param corr:
        Numpy 2-dim array with the correlation matrix for the assets.
        The element in the i'th row and j'th column is the correlation
        between assets i and j.

    :returns:
        Numpy array with the Full Exposure of each asset.
    """
    # Number of assets in the portfolio.
    n = len(weights)

    # Initialize an empty array for the results.
    full_exp = np.empty(shape=n, dtype=np.float64)

    # For each asset i in the portfolio.
    # Note the use of prange() instead of range() which instructs Numba Jit
    # to parallelize this loop, but only if @jit(parallel=True) was used,
    # otherwise this just becomes the ordinary Python range().
    for i in prange(n):
        # Portfolio weight of asset i.
        w_i = weights[i]

        # Initialize the sum of correlated exposures.
        sum_corr_exp = 0.0

        # For each other asset j in the portfolio.
        for j in range(n):
            # Portfolio weight of asset j.
            w_j = weights[j]

            # Correlation between assets i and j.
            c = corr[i, j]

            # Product of the two asset weights and their correlation.
            prod = w_i * w_j * c

            # If the product is positive then the correlation is deemed "bad"
            # and must be included in the calculation of the Full Exposure,
            # so the two portfolio weights can be adjusted accordingly.
            # This is explained in Section 8.3 of the paper referenced above.
            if prod > 0.0:
                # Multiply with the correlation again, because otherwise the
                # square-root calculated below would amplify the correlation.
                # Because this can result in a negative number, we also need
                # to take the absolute value.
                sum_corr_exp += np.abs(prod * c)

        # Calculate and save the Full Exposure for asset i.
        full_exp[i] = np.sign(w_i) * np.sqrt(sum_corr_exp)

    return full_exp


# Parallel Numba Jit version of the function `full_exposure`.
full_exposure_par = jit(full_exposure.py_func, parallel=True)


###############################################################################
# Mean Squared Error.

def mse_full_exposure(weights_new, weights_org, corr):
    """
    Mean Squared Error (MSE) between the original asset-weights
    and the Full Exposure of the new asset-weights.

    When the MSE value is zero, it means that the Full Exposure of
    the new asset-weights are equal to the original asset-weights.

    :param weights_org:
        Numpy array with the original asset-weights.

    :param weights_new:
        Numpy array with the new asset-weights.

    :param corr:
        Numpy 2-dim array for the correlation matrix between assets.

    :return:
        Float with the MSE value.
    """
    # Calculate the Full Exposure of the new asset-weights.
    full_exp = full_exposure(weights=weights_new, corr=corr)

    # Calculate the Mean Squared Error.
    mse = np.mean((full_exp - weights_org) ** 2)

    return mse


###############################################################################
# Adjust portfolio weights using custom algorithm.

@jit(parallel=False)
def _update_weights(weights_org, weights_new, corr, parallel=False):
    """
    Helper-function for the function `diversify_weights` which performs
    a single update of the portfolio weights, using the algorithm from
    Section 8.8 of the paper referenced above.

    WARNING: This should NOT be run in parallel with Numba Jit because there
    is a "race condition" in the for-loop that would corrupt the results.

    :param weights_org:
        Numpy array with the original portfolio weights.

    :param weights_new:
        Numpy array with the adjusted portfolio weights. It is updated inplace.

    :param corr:
        Numpy 2-dim array with the correlation matrix.

    :param parallel:
        Boolean whether to use the parallel (True) or serial (False) version
        of the function `full_exposure`.

    :return:
        Float with the max absolute difference between the Full Exposure
        and the original portfolio weights. This is used to abort the outer
        algorithm's for-loop when sufficiently good weights have been found.
    """
    # This could be implemented entirely with Numpy, but would need special
    # handling in case the Full Exposure is zero for some assets. This is a
    # slightly cleaner implementation that is also faster with Numba jit.

    # Calculate the Full Exposure for all the portfolio weights.
    if parallel:
        # Parallel execution.
        full_exp = full_exposure_par(weights=weights_new, corr=corr)
    else:
        # Serial execution.
        full_exp = full_exposure(weights=weights_new, corr=corr)

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


def diversify_weights(weights_org, corr, weights_guess=None, fix_input=True,
                      parallel=False, log=None, max_iter=100, tol=1e-3):
    """
    Find new asset-weights that minimize the Mean Squared Error (MSE) between
    the original asset-weights and the Full Exposure of the new asset-weights,
    using a custom and extremely fast algorithm for this particular problem.

    For a portfolio of 1000 assets it only takes 20 milli-seconds to run this,
    depending on the CPU, arguments, and the weights and correlation matrix.
    Removing some of the options and overhead in the function can significantly
    improve the speed. But Numba Jit cannot improve the speed of this function.

    The diversification algorithm is explained in Section 8 of the paper:
    - M.E.H. Pedersen, "Simple Portfolio Optimization That Works!", 2021.
      https://ssrn.com/abstract=3942552
      https://github.com/Hvass-Labs/Finance-Papers

    :param weights_org:
        Array with the originally desired asset-weights for the portfolio.
        These can be either positive or negative and they need not sum to 1.
        This data can either be a Pandas Series or Numpy array.

    :param corr:
        Matrix with the correlations between assets.
        This can either be a Pandas DataFrame or Numpy array.

    :param weights_guess:
        Array with a starting guess for the adjusted portfolio weights.
        If you are calling this function with `weights_org` and `corr` being
        nearly identical on each call, then you might save computation time
        by passing the last weights that were output from this function as
        the arg `weights_guess` the next time you call this function. This
        may reduce the number of iterations needed for convergence.

    :param fix_input:
        Boolean whether to repair input by filling NaN-values (Not-a-Number)
        in `weights_org` and `weights_guess`, and various repairs to `corr`.

    :param parallel:
        Boolean whether to use the parallel (True) or serial (False) version
        of the function `full_exposure`. Whether the parallel version is faster
        depends on your problem, so you should test which is fastest for you.

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
    # If using Pandas data-types, ensure their index and column names match.
    _check_pandas_index(weights_org=weights_org, corr=corr,
                        weights_guess=weights_guess)

    # Convert weights_org from Pandas to Numpy.
    if isinstance(weights_org, (pd.Series, pd.DataFrame)):
        # Save the Pandas index for later use with the return-data.
        index = weights_org.index

        # Convert Pandas to Numpy. This may or may not be a copy of the data.
        # Note: Flatten is necessary if it is a Pandas DataFrame.
        weights_org = weights_org.to_numpy().flatten()
    else:
        # This is also used to indicate that the input was NOT Pandas data.
        index = None

    # Convert weights_guess from Pandas to Numpy.
    if isinstance(weights_guess, (pd.Series, pd.DataFrame)):
        # This may or may not be a copy of the data.
        # Note: Flatten is necessary if it is a Pandas DataFrame.
        weights_guess = weights_guess.to_numpy().flatten()

    # Convert correlation matrix from Pandas to Numpy.
    if isinstance(corr, pd.DataFrame):
        # This may or may not be a copy of the data.
        corr = corr.to_numpy()

    # Ensure the weights and correlation-matrix are valid.
    if fix_input:
        # This copies the data.
        weights_org, corr, weights_guess = \
            _fix_input(weights_org=weights_org, corr=corr,
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
        max_abs_dif = _update_weights(weights_org=weights_org,
                                      weights_new=weights_new,
                                      corr=corr, parallel=parallel)

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

def log_to_dataframe(weights_org, corr, log):
    """
    Convert the log from `diversify_weights` to a Pandas DataFrame which shows
    the iterations of the adjusted portfolio weights and their Full Exposure.

    :param weights_org:
        Numpy array with the originally desired portfolio weights.

    :param corr:
        Numpy 2-dim array with the correlation matrix.

    :param log:
        List of numpy arrays with portfolio weights. This is obtained by first
        passing the list as the `log` arg in the `diversify_weights` function.

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
        fe = full_exposure(weights=weights_new, corr=corr)

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
