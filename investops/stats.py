###############################################################################
#
# Various statistical functions.
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
from numba import jit

###############################################################################

@jit
def weighted_mean_std(x, weights):
    """
    Calculate the mean and standard deviation for a weighted array of numbers.

    :param x: Numpy array of numbers.
    :param weights: Numpy array of weights for the numbers.
    :return:
        - Numpy array with the weighted mean.
        - Numpy array with the weighted std.dev.
    """
    # Weighted mean.
    mean = np.average(x, weights=weights)

    # Weighted variance.
    var = np.average((x - mean) ** 2, weights=weights)

    # Weighted std.dev.
    std = np.sqrt(var)

    return mean, std

###############################################################################
