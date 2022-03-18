###############################################################################
#
# Various metric functions.
#
# These may be available in other libraries such as scikit-learn, but it
# would make a big Python package dependency for just a few small functions.
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

###############################################################################

def r_squared(y_true, y_pred):
    """
    Calculate the Coefficient of Determination R^2 for measuring the
    Goodness of Fit between the true and predicted data.

    An R^2 value of one means there is a perfect fit and the predicted
    data explains all the variance in the data. An R^2 value of zero means
    the predicted data does not explain any of the variance in the data.

    Note: If the predictive model is non-linear, then the R^2 can become
    negative if the model fits poorly on data with a large variance.

    :param y_true:
        Numpy array with the observed data-values.

    :param y_pred:
        Numpy array with the predicted data-values from some model.

    :return:
        Float with the R^2 value.
    """
    # Errors between the observed and predicted data.
    err_pred = (y_true - y_pred) ** 2

    # Baseline errors between the observed data and its own mean.
    err_baseline = (y_true - np.mean(y_true)) ** 2

    # Sum of Squared Errors (SSE) for the predictive errors.
    sse = np.sum(err_pred)

    # Sum of Squared Errors (SST) for the baseline errors.
    sst = np.sum(err_baseline)

    # The R^2 value.
    r2 = 1.0 - sse / sst

    return r2

###############################################################################
