###############################################################################
#
# Various utility functions.
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

def dropna(x):
    """
    Remove NaN (Not-a-Number) elements from a 1-dim Numpy array.

    :param x: Numpy array.
    :return: Numpy array.
    """
    return x[~np.isnan(x)]

###############################################################################
