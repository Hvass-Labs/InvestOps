###############################################################################
#
# Utils for testing.
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

from numpy.testing import assert_array_compare
import operator

###############################################################################


def assert_array_less_equal(x, y, err_msg='', verbose=True):
    """
    Raises an AssertionError if two array_like objects are not ordered by
    less-or-equal than.

    This a simple modification of `assert_array_less` from `numpy.testing`.
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    assert_array_compare(operator.__le__, x, y, err_msg=err_msg,
                         verbose=verbose,
                         header='Arrays are not less-or-equal-ordered',
                         equal_inf=False)


###############################################################################
