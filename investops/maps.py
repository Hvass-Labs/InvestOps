###############################################################################
#
# Functions and classes for mathematical mappings.
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

from numbers import Number

###############################################################################


class LinearMap:
    """
    Linear map from input `x` to output `y` with optional boundaries.

    if x < x1 and y1_fill is not None:
        # Left boundary.
        y = y1_fill
    elif x > x2 and y2_fill is not None:
        # Right boundary.
        y = y2_fill
    else:
        # Linear mapping between boundaries.
        # x == x1 maps to y == y1
        # x == x2 maps to y == y2
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        y = a * x + b
    """
    def __init__(self, x1, y1, x2, y2, y1_fill=None, y2_fill=None):
        """
        :param x1: Left point for linear mapping (x1, y1)
        :param y1: Left point for linear mapping (x1, y1)
        :param x2: Right point for linear mapping (x2, y2)
        :param y2: Right point for linear mapping (x2, y2)
        :param y1_fill:
            Left boundary for linear mapping: If `x < x1` then `y = y1_fill`.
            If `None` then continue the linear mapping beyond `x1`.
        :param y2_fill:
            Right boundary for linear mapping: If `x > x2` then `y = y2_fill`.
            If `None` then continue the linear mapping beyond `x2`.
        :raises:
            `TypeError`: An argument is of the wrong type.
            `ValueError`: An argument has an invalid value.
        """
        # Check argument types.
        number_types = (float, int)
        if not isinstance(x1, number_types):
            raise TypeError('Arg `x1` is not a float or int.')
        if not isinstance(x2, number_types):
            raise TypeError('Arg `x2` is not a float or int.')
        if not isinstance(y1, number_types):
            raise TypeError('Arg `y1` is not a float or int.')
        if not isinstance(y2, number_types):
            raise TypeError('Arg `y2` is not a float or int.')
        if y1_fill is not None and not isinstance(y1_fill, number_types):
            raise TypeError('Arg `y1_fill` is not a float or int.')
        if y2_fill is not None and not isinstance(y2_fill, number_types):
            raise TypeError('Arg `y2_fill` is not a float or int.')

        # Check arguments are valid.
        if x1 > x2:
            msg = f'Arg `x1=={x1}` is higher than `x2=={x2}`'
            raise ValueError(msg)

        # Copy some arguments to self for later use.
        self._x1 = x1
        self._x2 = x2
        self._y1_fill = y1_fill
        self._y2_fill = y2_fill

        # Calculate coefficients for the linear mapping.
        self._a = (y2 - y1) / (x2 - x1)
        self._b = y1 - self._a * x1

    def __call__(self, x):
        """
        :param x:
            Either a single float, int or other type of number,
            or a Numpy or Pandas array of numbers. Data-type is NOT checked!

        :return:
            Linear mapping of the same type as input `x`.
        """
        # Calculate the linear map for all input-values.
        y = self._a * x + self._b

        # Copy variables from self for easy reference.
        y1_fill = self._y1_fill
        y2_fill = self._y2_fill

        # Limit the values beyond the left boundary?
        if y1_fill is not None:
            # Boolean condition whether the input is beyond the boundary.
            # This works for both scalar values and arrays.
            cond = (x < self._x1)

            # Set the relevant output-numbers to the fill-number.
            if isinstance(x, Number):
                # Input is a single scalar number.
                if cond:
                    y = y1_fill
            else:
                # Input is assumed to be a Numpy or Pandas array of numbers.
                y[cond] = y1_fill

        # Limit the values beyond the right boundary?
        if self._y2_fill is not None:
            # Boolean condition whether the input is beyond the boundary.
            # This works for both scalar values and arrays.
            cond = (x > self._x2)

            # Set the relevant output-numbers to the fill-number.
            if isinstance(x, Number):
                if cond:
                    # Input is a single scalar number.
                    y = y2_fill
            else:
                # Input is assumed to be a Numpy or Pandas array of numbers.
                y[cond] = y2_fill

        return y


###############################################################################
