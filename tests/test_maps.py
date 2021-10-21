###############################################################################
#
# Tests for investops.maps
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

from investops.maps import LinearMap
import numpy as np
from numpy.testing import assert_allclose

###############################################################################
# LinearMap()


def test_LinearMap():
    """Test investops.maps.LinearMap class."""
    # Test 1
    map = LinearMap(x1=0.1, y1=0.2, x2=0.8, y2=0.9, y1_fill=0.05, y2_fill=0.95)
    x = np.arange(0.0, 1.0, step=0.05)
    y = map(x)
    assert_allclose(y, [0.05, 0.05, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
                        0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.95, 0.95])

    # Test 2
    map = LinearMap(x1=-0.5, y1=-0.4, x2=0.5, y2=0.8, y1_fill=0.3, y2_fill=0.9)
    x = np.arange(-1.0, 1.0, step=0.05)
    y = map(x)
    assert_allclose(y, [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, -0.4,
                        -0.34, -0.28, -0.22, -0.16, -0.1, -0.04, 0.02, 0.08,
                        0.14, 0.2, 0.26, 0.32, 0.38, 0.44, 0.5, 0.56, 0.62,
                        0.68, 0.74, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                        0.9, 0.9])


###############################################################################
