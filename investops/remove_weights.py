###############################################################################
#
# Functions for removing zero/small portfolio weights and their correlations.
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

###############################################################################
# Common helper-functions.


def _check_type(weights, corr=None):
    """
    Check that the arguments have the correct type.

    :param weights: Pandas Series with the portfolio weights.
    :param corr: Pandas DataFrame with the correlation matrix.
    :raises: `TypeError` if the args have incorrect types.
    :return: None
    """
    # Check the type of the weights.
    if not isinstance(weights, pd.Series):
        msg = 'Argument \'weights\' must be a Pandas Series.'
        raise TypeError(msg)

    # Check the type of the correlation matrix.
    if corr is not None and not isinstance(corr, pd.DataFrame):
        msg = 'Argument \'corr\' must be a Pandas DataFrame.'
        raise TypeError(msg)


def _remove_weights(mask, weights, corr=None):
    """
    Helper-function for removing portfolio weights according to a boolean mask.

    :param mask: Boolean mask for the weights to keep (True) or remove (False).
    :param weights: Pandas Series with the portfolio weights.
    :param corr: Pandas DataFrame with the correlation matrix.
    :return:
        If arg `corr` is not `None` then return a tuple with the new
        weights and correlations. Otherwise, just return the new weights.
    """

    # Only use the weights where the boolean mask is True.
    weights_new = weights[mask]

    if corr is not None:
        # Use the index of the weights (e.g. stock-tickers) instead of the
        # boolean mask, because then it also works for correlation matrices
        # that have more assets than just those in the weights. This also
        # sorts the rows and columns of the corr matrix to match the weights.
        index = weights_new.index

        # Only use the relevant elements of the correlation matrix.
        corr_new = corr.loc[index, index]

        # Return both the new weights and correlations.
        return weights_new, corr_new
    else:
        # Only return the new weights.
        return weights_new


###############################################################################


def remove_zero_weights(weights, corr=None):
    """
    Remove the portfolio weights that are zero (both 0.0 and -0.0),
    and also remove their correlations if the arg `corr` is not `None`.

    If you also want to remove weights that are `NaN` (Not-a-Number) then you
    call the function `dropna()` on the `weights` before calling this function.

    :param weights: Pandas Series with the portfolio weights.
    :param corr: Pandas DataFrame with the correlation matrix.
    :return:
        If arg `corr` is not `None` then return a tuple with the new
        weights and correlations. Otherwise, just return the new weights.
    """
    # Check the arg types are correct.
    _check_type(weights=weights, corr=corr)

    # Boolean mask for keeping only the non-zero portfolio weights.
    mask = (weights != 0.0)

    return _remove_weights(mask=mask, weights=weights, corr=corr)


def remove_low_weights(weights, corr=None, threshold=0.0):
    """
    Remove the portfolio weights whose absolute value is below the threshold,
    and also remove their correlations if the arg `corr` is not `None`.

    If you also want to remove weights that are `NaN` (Not-a-Number) then you
    call the function `dropna()` on the `weights` before calling this function.

    :param weights: Pandas Series with the portfolio weights.
    :param corr: Pandas DataFrame with the correlation matrix.
    :param threshold: Positive float with threshold for the portfolio weights.
    :return:
        If arg `corr` is not `None` then return a tuple with the new
        weights and correlations. Otherwise, just return the new weights.
    """
    # Check the arg types are correct.
    _check_type(weights=weights, corr=corr)

    # Check other args.
    if threshold < 0.0:
        msg = 'Argument \'threshold\' must be a positive number.'
        raise ValueError(msg)

    # Boolean mask for keeping only the relevant portfolio weights.
    mask = (np.abs(weights) >= threshold)

    return _remove_weights(mask=mask, weights=weights, corr=corr)


###############################################################################
