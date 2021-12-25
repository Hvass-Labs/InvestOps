###############################################################################
#
# Tests for investops.remove_weights
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

from investops.remove_weights import remove_low_weights, remove_zero_weights
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal

###############################################################################
# Test-case inputs.

# Input for the portfolio weights.
_weights = pd.Series(dict(MSFT=0.1, AAPL=np.nan, AMZN=0.0, BBBY=-0.3))

# Input for the correlation matrix.
_corr = \
    {'MSFT': {'MSFT': 1.0, 'AAPL': 0.5, 'AMZN': 0.3, 'BBBY': 0.9},
     'AAPL': {'MSFT': 0.5, 'AAPL': 1.0, 'AMZN': -0.1, 'BBBY': -0.3},
     'AMZN': {'MSFT': 0.3, 'AAPL': -0.1, 'AMZN': 1.0, 'BBBY': 0.4},
     'BBBY': {'MSFT': 0.9, 'AAPL': -0.3, 'AMZN': 0.4, 'BBBY': 1.0}}
_corr = pd.DataFrame(_corr)

###############################################################################
# remove_zero_weights()

def test_remove_zero_weights():
    """Test investops.remove_weights.remove_zero_weights()"""

    # Test 1
    weights_out, corr_out = remove_zero_weights(weights=_weights, corr=_corr)
    weights_chk = pd.Series(dict(MSFT=0.1, AAPL=np.nan, BBBY=-0.3))
    corr_chk = \
        {'MSFT': {'MSFT': 1.0, 'AAPL': 0.5, 'BBBY': 0.9},
         'AAPL': {'MSFT': 0.5, 'AAPL': 1.0, 'BBBY': -0.3},
         'BBBY': {'MSFT': 0.9, 'AAPL': -0.3, 'BBBY': 1.0}}
    corr_chk = pd.DataFrame(corr_chk)
    assert_series_equal(weights_out, weights_chk)
    assert_frame_equal(corr_out, corr_chk)

    # Test 2
    weights_out, corr_out = remove_zero_weights(weights=_weights.dropna(),
                                                corr=_corr)
    weights_chk = pd.Series(dict(MSFT=0.1, BBBY=-0.3))
    corr_chk = \
        {'MSFT': {'MSFT': 1.0, 'BBBY': 0.9},
         'BBBY': {'MSFT': 0.9, 'BBBY': 1.0}}
    corr_chk = pd.DataFrame(corr_chk)
    assert_series_equal(weights_out, weights_chk)
    assert_frame_equal(corr_out, corr_chk)

###############################################################################
# remove_low_weights()

def test_remove_low_weights():
    """Test investops.remove_weights.remove_low_weights()"""

    # Test 1
    # Note: This function automatically removes weights that are NaN. But in
    # practice it is recommended you call _weights.dropna() for clarity.
    weights_out, corr_out = remove_low_weights(weights=_weights, corr=_corr,
                                               threshold=0.1)
    weights_chk = pd.Series(dict(MSFT=0.1, BBBY=-0.3))
    corr_chk = \
        {'MSFT': {'MSFT': 1.0, 'BBBY': 0.9},
         'BBBY': {'MSFT': 0.9, 'BBBY': 1.0}}
    corr_chk = pd.DataFrame(corr_chk)
    assert_series_equal(weights_out, weights_chk)
    assert_frame_equal(corr_out, corr_chk)

###############################################################################
