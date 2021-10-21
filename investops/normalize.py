###############################################################################
#
# Functions for normalizing portfolio weights.
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
import numba
from numba import jit

###############################################################################
# Common helper-functions.

def _convert_input_weights(weights):
    """
    Convert the input weights from Numpy or Pandas to a 2-dim Numpy array.

    :param weights:
        Array with portfolio weights. Either a Numpy Array, Pandas Series, or
        Pandas DataFrame. Rows are for time-steps, columns are for assets.

    :raises:
        `TypeError` if the `weights` arg is of the wrong data-type.

    :return:
        Numpy 2-dim array with portfolio weights.
    """
    # Convert the weights data to a 2-dim Numpy array.
    if isinstance(weights, pd.Series):
        # Convert Pandas Series to 2-dim Numpy array.
        weights_np = weights.to_numpy()[np.newaxis, :]
    elif isinstance(weights, pd.DataFrame):
        # Convert Pandas DataFrame to 2-dim Numpy array.
        weights_np = weights.to_numpy()
    elif isinstance(weights, np.ndarray):
        if len(weights.shape) == 1:
            # Convert Numpy 1-dim array to 2-dim Numpy array.
            weights_np = weights[np.newaxis, :]
        else:
            # Use the Numpy 2-dim array as it is.
            weights_np = weights
    else:
        # Raise exception for invalid data-type.
        msg = 'Argument `weights` must be either Numpy Array, ' \
              'Pandas Series or DataFrame.'
        raise TypeError(msg)

    return weights_np


def _convert_output_weights(weights, weights_norm, cash):
    """
    Convert the normalized weights to the same data-type as the input weights.

    :param weights:
        Array with the input weights, only used to infer its data-type.

    :param weights_norm:
        Numpy 2-dim array with the normalized weights.

    :param cash:
        Numpy 1-dim array with the cash-positions.

    :return:
        Tuple with the normalized weights and cash as the proper data-types.
    """
    # Convert the result to the same data-type as the input.
    if isinstance(weights, pd.Series):
        # Convert Numpy 2-dim array to Pandas Series.
        weights_norm = pd.Series(data=weights_norm[0], index=weights.index)
        # Convert Numpy 1-dim array to scalar float.
        cash = cash[0]
    elif isinstance(weights, pd.DataFrame):
        # Convert Numpy 2-dim array to Pandas DataFrame.
        weights_norm = pd.DataFrame(data=weights_norm, index=weights.index,
                                    columns=weights.columns)
        # Convert Numpy 1-dim array to Pandas Series.
        cash = pd.Series(data=cash, index=weights.index)
    elif isinstance(weights, np.ndarray) and len(weights.shape) == 1:
        # Convert Numpy 2-dim array to Numpy 1-dim array.
        weights_norm = weights_norm[0, :]
        # Convert Numpy 1-dim array to scalar float.
        cash = cash[0]

    return weights_norm, cash


###############################################################################
# Functions for normalizing only positive weights.

@jit
def _normalize_weights_pos_jit(weights, limit_pos):
    """
    Fast implementation of the function `normalize_weights_pos` with Numba Jit.

    It is possible to implement this directly using Pandas, but even with a
    clever implementation it is harder to understand and runs much slower.

    :param weights:
        Numpy array with portfolio weights. This MUST be 2-dimensional!
        Rows are for time-steps, columns are for assets.

    :param limit_pos:
        Float with the maximum allowed sum of positive weights.

    :return:
        Tuple with the following data:
        - Numpy array 2-dim with the normalized weights.
        - Numpy array 1-dim with the cash-position for each time-step.
    """
    # Number of rows (time-steps) and number of columns (assets).
    num_rows, num_cols = weights.shape

    # Initialize the arrays for the normalized weights and cash-position.
    weights_norm = weights.copy()
    cash = np.zeros(num_rows, dtype=numba.float32)

    # For each row (time-step).
    for i in range(num_rows):
        # Initialize the sum of positive and negative weights for this row.
        weight_sum = 0.0

        # For each column (asset).
        for j in range(num_cols):
            # Get the weight for this row and column.
            w_ij = weights[i, j]

            # Accumulate the sum of weights for this row.
            # Note: We do NOT check if the weights are positive!
            weight_sum += w_ij

        if weight_sum > limit_pos:
            # Scale for all weights in this row to bring their sum down.
            scale = limit_pos / weight_sum

            # For each column (asset).
            for j in range(num_cols):
                # Update the weight with the scale.
                weights_norm[i, j] *= scale
        else:
            # This row was not scaled.
            scale = 1.0

        # Calculate and set the cash-position for this row (time-step).
        # Note: This can be negative if limit_pos > 1.0
        cash[i] = 1.0 - scale * weight_sum

    return weights_norm, cash


def normalize_weights_pos(weights, limit_pos=1.0):
    """
    Normalize portfolio weights so their sum does not exceed some limit. This
    function is only intended for positive (aka. "long") portfolio weights,
    and for maximum speed it does NOT check if the input weights are positive!

    If you are not sure whether some weights might be negative, you either need
    to clip the input-data to ensure it is positive, or you should use the
    other function `normalize_weights` which can also handle negative weights.

    This function is actually only about 20% faster than `normalize_weights`.

    :param weights:
        Array with portfolio weights. Either a Numpy Array, Pandas Series, or
        Pandas DataFrame. Rows are for time-steps, columns are for assets.

    :param limit_pos:
        Float with the maximum allowed sum of positive weights. Examples:
        Set this to 1.0 if you do not want to invest for borrowed money.
        Or set this to 0.8 if you only want to invest 80% of your money.
        Or set this to 1.2 if you want to invest for 20% borrowed money.

    :return:
        Tuple with the following data:
        - Array with the normalized weights. Same type and shape as `weights`.
        - Cash-position for each time-step, which can be negative if
          `limit_pos > 1.0` because you are investing for borrowed money.
          If `weights` is Pandas DataFrame, then this is a Pandas Series.
          If 'weights' is Numpy 2-dim array, then this is a 1-dim array.
          If `weights` is a Pandas Series or Numpy 1-dim array,
          then this is a float value.
    """
    # Convert input weights to Numpy 2-dim array.
    weights_np = _convert_input_weights(weights)

    # Run the Jit-optimized function to normalize the weights.
    weights_norm, cash = _normalize_weights_pos_jit(weights=weights_np,
                                                    limit_pos=limit_pos)

    # Convert output weights to same data-type as input weights.
    weights_norm, cash = _convert_output_weights(weights=weights, cash=cash,
                                                 weights_norm=weights_norm)

    return weights_norm, cash


###############################################################################
# Functions for normalizing both positive and negative weights.

@jit
def _normalize_weights_jit(weights, limit_pos, limit_neg, max_ratio):
    """
    Fast implementation of the function `normalize_weights` with Numba Jit.

    It is possible to implement this directly using Pandas, but even with a
    clever implementation, it is much harder to understand the code, and it
    runs about 1000x slower! The only drawback to this implementation, is
    that Numba Jit only supports Numpy arrays, so we must split the function
    into two parts: This function that does the actual computation, and another
    wrapper-function that can also handle Pandas Series and DataFrames.

    :param weights:
        Numpy array with portfolio weights. This MUST be 2-dimensional!
        Rows are for time-steps, columns are for assets.

    :param limit_pos:
        Float with the maximum allowed sum of positive weights.

    :param limit_neg:
        Float with the minimum allowed sum of negative weights.

    :param max_ratio:
        Float with maximum allowed ratio between positive and negative weights.

    :return:
        Tuple with the following data:
        - Numpy array 2-dim with the normalized weights.
        - Numpy array 1-dim with the cash-position for each time-step.
    """
    # Number of rows (time-steps) and number of columns (assets).
    num_rows, num_cols = weights.shape

    # Initialize the arrays for the normalized weights and cash-position.
    weights_norm = weights.copy()
    cash = np.zeros(num_rows, dtype=numba.float32)

    # For each row (time-step).
    for i in range(num_rows):
        # Initialize the sums of positive and negative weights for this row.
        sum_pos = 0.0
        sum_neg = 0.0

        # For each column (asset).
        for j in range(num_cols):
            # Get the weight for this row and column.
            w_ij = weights[i, j]

            # Accumulate sums of positive and negative weights for this row.
            if w_ij >= 0.0:
                sum_pos += w_ij
            else:
                sum_neg += w_ij

        # Scale used if the sum of positive weights exceeds its limit.
        if sum_pos > limit_pos:
            scale_pos = limit_pos / sum_pos
        else:
            scale_pos = 1.0

        # Scale used if the sum of negative weights exceeds its limit.
        if sum_neg < limit_neg:
            scale_neg = limit_neg / sum_neg
        else:
            scale_neg = 1.0

        # Adjust the scale for negative weights, if the sum of negative weights
        # exceed the sum of positive weights by the given ratio.
        if -sum_neg * scale_neg > max_ratio * sum_pos * scale_pos:
            scale_neg = max_ratio * sum_pos * scale_pos / -sum_neg

        # Scale this row if necessary.
        if scale_pos != 1.0 or scale_neg != 1.0:
            # For each column (asset).
            for j in range(num_cols):
                # Use the positive or negative scale for this weight.
                scale = scale_pos if weights[i, j] >= 0.0 else scale_neg

                # Update the weight with the scale.
                weights_norm[i, j] *= scale

        # Calculate and set the cash-position for this row (time-step).
        # Note: This can be negative if limit_pos > 1.0
        cash[i] = 1.0 - scale_pos * sum_pos

    return weights_norm, cash


def normalize_weights(weights, limit_pos=1.0, limit_neg=0.0, max_ratio=0.0):
    """
    Normalize portfolio weights so their sum does not exceed some limit.
    This can be used for "long", "short" and both "long and short" portfolios.

    For each time-step, the sums are calculated independently for the positive
    and negative weights. If the sums exceed their limits, then the weights are
    scaled to bring their sums back to their limits. Furthermore, the sum of
    negative weights is not allowed to exceed the sum of positive weights by
    the given `max_ratio`.

    NOTE: You should check that this normalization is consistent with your
    investment broker's requirements for "short" vs. "long" positions.

    :param weights:
        Array with portfolio weights. Either a Numpy Array, Pandas Series, or
        Pandas DataFrame. Rows are for time-steps, columns are for assets.

    :param limit_pos:
        Float with the maximum allowed sum of positive weights. Examples:
        Set this to 1.0 if you do not want to invest for borrowed money.
        Or set this to 0.8 if you only want to invest 80% of your money.
        Or set this to 1.2 if you want to invest for 20% borrowed money.

    :param limit_neg:
        Float with the minimum allowed sum of negative weights. Example:
        Set this to -0.5 if you only want to "short" for 50% of your money.

    :param max_ratio:
        Float with maximum allowed ratio between positive and negative weights.
        Example: Set this to 0.2 if the sum of negative weights must not exceed
        20% of the sum of positive weights.

    :return:
        Tuple with the following data:
        - Array with the normalized weights. Same type and shape as `weights`.
        - Cash-position for each time-step, which can be negative if
          `limit_pos > 1.0` because you are investing for borrowed money.
          If `weights` is Pandas DataFrame, then this is a Pandas Series.
          If 'weights' is Numpy 2-dim array, then this is a 1-dim array.
          If `weights` is a Pandas Series or Numpy 1-dim array,
          then this is a float value.
    """
    # Convert input weights to Numpy 2-dim array.
    weights_np = _convert_input_weights(weights)

    # Run the Jit-optimized function to normalize the weights.
    weights_norm, cash = _normalize_weights_jit(weights=weights_np,
                                                limit_pos=limit_pos,
                                                limit_neg=limit_neg,
                                                max_ratio=max_ratio)

    # Convert output weights to same data-type as input weights.
    weights_norm, cash = _convert_output_weights(weights=weights, cash=cash,
                                                 weights_norm=weights_norm)

    return weights_norm, cash


###############################################################################
