###############################################################################
#
# Tests for investops.rel_change
#
# We use a small number of test-cases that have been manually calculated so we
# know what the correct results should be. We do not test long time-series
# e.g. with many years of share-price data. Therefore we also do not test if
# the annualized calculation is correct. We do check that the geometric mean
# and log-transform of the time-series data is done correctly.
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

from investops.rel_change import rel_change, mean_rel_change
import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal, assert_frame_equal

###############################################################################
# Helper-functions.

def _geo(df, n):
    """Geometric mean of relative changes in a time-series."""
    return (df + 1.0) ** (1/n) - 1.0


def _geo_log(df, n=1):
    """Geometric mean of log-transformed relative changes in a time-series."""
    return np.log(df + 1.0) / n


def _mean_std(list_df, freq='D'):
    """
    Calculate the element-wise mean and std.dev.
    for a list of Pandas DataFrames or Series.

    :param list_df: List of Pandas DataFrames or Series.
    :param freq: Frequency for the index of the resulting DataFrames.
    :return:
        - Pandas DataFrame or Series with the mean values.
        - Pandas DataFrame or Series with the std.dev. values.
    """
    # Concatenate all columns of the DataFrames.
    df_concat = pd.concat(list_df)

    # Group-by the index-values.
    df_groupby = df_concat.groupby(df_concat.index)

    # Calculate the mean, and set result to NaN if contains NaN.
    df_mean = df_groupby.agg(lambda x: x.mean(skipna=False))

    # Calculate the std.dev., and set result to NaN if contains NaN.
    # The ddof (delta degrees of freedom) matches that used by mean_rel_change.
    df_std = df_groupby.agg(lambda x: x.std(skipna=False, ddof=0))

    # Set the frequency of the indices, otherwise the assert-equal will fail.
    df_mean.index.freq = freq
    df_std.index.freq = freq

    return df_mean, df_std

###############################################################################
# Test-case inputs.

# ORIGINAL time-series data used as input, this could be e.g. share-prices.
_x1 = np.array([0.5, 1.0, 1.5, 1.8, 2.0, 1.8, 1.2])
_x2 = np.array([3.0, 2.0, 0.8, 0.6, 0.4, 0.8, 0.2])
_x3 = np.array([1.0, 1.2, 0.8, 0.6, 1.2, 1.8, 0.9])
_x = np.vstack([_x1, _x2, _x3]).T

# Correct values for the relative change: 1 day, FUTURE.
# For example: _y2_f1[0] = _x2[1] / _x2[0] - 1.0 = 2.0 / 3.0 - 1.0 = -0.333333
_y1_f1 = np.array([1.0, 0.5, 0.2, 0.11111111, -0.1, -0.33333333, np.nan])
_y2_f1 = np.array([-0.33333333, -0.6, -0.25, -0.33333333, 1.0, -0.75, np.nan])
_y3_f1 = np.array([0.2, -0.33333333, -0.25, 1.0, 0.5, -0.5, np.nan])
_y_f1 = np.vstack([_y1_f1, _y2_f1, _y3_f1]).T

# Correct values for the relative change: 2 day, FUTURE.
# For example: _y2_f2[0] = _x2[2] / _x2[0] - 1.0 = 0.8 / 3.0 - 1.0 = -0.733333
_y1_f2 = np.array([2.0, 0.8, 0.33333333, 0.0, -0.4, np.nan, np.nan])
_y2_f2 = np.array([-0.73333333, -0.7, -0.5, 0.33333333, -0.5, np.nan, np.nan])
_y3_f2 = np.array([-0.2, -0.5, 0.5, 2.0, -0.25, np.nan, np.nan])
_y_f2 = np.vstack([_y1_f2, _y2_f2, _y3_f2]).T

# Correct values for the relative change: 3 day, FUTURE.
# For example: _y2_f3[0] = _x2[3] / _x2[0] - 1.0 = 0.6 / 3.0 - 1.0 = -0.8
_y1_f3 = np.array([2.6, 1.0, 0.2, -0.33333333, np.nan, np.nan, np.nan])
_y2_f3 = np.array([-0.8, -0.8, 0.0, -0.66666667, np.nan, np.nan, np.nan])
_y3_f3 = np.array([-0.4, 0.0, 1.25, 0.5, np.nan, np.nan, np.nan])
_y_f3 = np.vstack([_y1_f3, _y2_f3, _y3_f3]).T

# Correct values for the relative change: 1 day, PAST.
_y1_p1 = np.roll(_y1_f1, shift=1)
_y2_p1 = np.roll(_y2_f1, shift=1)
_y3_p1 = np.roll(_y3_f1, shift=1)
_y_p1 = np.vstack([_y1_p1, _y2_p1, _y3_p1]).T

# Correct values for the relative change: 2 day, PAST.
_y1_p2 = np.roll(_y1_f2, shift=2)
_y2_p2 = np.roll(_y2_f2, shift=2)
_y3_p2 = np.roll(_y3_f2, shift=2)
_y_p2 = np.vstack([_y1_p2, _y2_p2, _y3_p2]).T

# Correct values for the relative change: 3 day, PAST.
_y1_p3 = np.roll(_y1_f3, shift=3)
_y2_p3 = np.roll(_y2_f3, shift=3)
_y3_p3 = np.roll(_y3_f3, shift=3)
_y_p3 = np.vstack([_y1_p3, _y2_p3, _y3_p3]).T

# Index with dates for the Pandas data.
_index = pd.date_range(start='2022-01-01', periods=len(_x), freq='D')

# ORIGINAL input time-series as Pandas data.
# Pandas Series.
_ser_x1 = pd.Series(_x1, index=_index)
_ser_x2 = pd.Series(_x2, index=_index)
_ser_x3 = pd.Series(_x3, index=_index)
# Pandas DataFrame.
_df_x1 = pd.DataFrame(_x1, index=_index)
_df_x2 = pd.DataFrame(_x2, index=_index)
_df_x3 = pd.DataFrame(_x3, index=_index)
_df_x = pd.DataFrame(_x, index=_index)

# Pandas Series.
# Correct values for the relative change: 1 day, FUTURE.
_ser_y1_f1 = pd.Series(_y1_f1, index=_index)
_ser_y2_f1 = pd.Series(_y2_f1, index=_index)
_ser_y3_f1 = pd.Series(_y3_f1, index=_index)

# Pandas Series.
# Correct values for the relative change: 1 day, PAST.
_ser_y1_p1 = pd.Series(_y1_p1, index=_index)
_ser_y2_p1 = pd.Series(_y2_p1, index=_index)
_ser_y3_p1 = pd.Series(_y3_p1, index=_index)

# Pandas DataFrame with one column.
# Correct values for the relative change: 1 day, FUTURE.
_df_y1_f1 = pd.DataFrame(_y1_f1, index=_index)
_df_y2_f1 = pd.DataFrame(_y2_f1, index=_index)
_df_y3_f1 = pd.DataFrame(_y3_f1, index=_index)

# Pandas DataFrame with 3 columns.
# Correct values for the relative change: 1, 2, and 3 day, FUTURE.
_df_y_f1 = pd.DataFrame(_y_f1, index=_index)
_df_y_f2 = pd.DataFrame(_y_f2, index=_index)
_df_y_f3 = pd.DataFrame(_y_f3, index=_index)

# Pandas DataFrame with 3 columns.
# Correct values for the relative change: 1, 2, and 3 day, PAST.
_df_y_p1 = pd.DataFrame(_y_p1, index=_index)
_df_y_p2 = pd.DataFrame(_y_p2, index=_index)
_df_y_p3 = pd.DataFrame(_y_p3, index=_index)

# Correct values for mean and std.dev. relative change: 1-2 days, FUTURE.
_list_y_f1_2 = [_df_y_f1, _df_y_f2]
_df_y_f1_2_mean, _df_y_f1_2_std = _mean_std(_list_y_f1_2)

# Correct values for mean and std.dev. relative change: 1-2 days, PAST.
_list_y_p1_2 = [_df_y_p1, _df_y_p2]
_df_y_p1_2_mean, _df_y_p1_2_std = _mean_std(_list_y_p1_2)

# Correct values for mean and std.dev. relative change: 1-3 days, FUTURE.
_list_y_f1_3 = [_df_y_f1, _df_y_f2, _df_y_f3]
_df_y_f1_3_mean, _df_y_f1_3_std = _mean_std(_list_y_f1_3)

# Correct values for mean and std.dev. relative change: 1-3 days, PAST.
_list_y_p1_3 = [_df_y_p1, _df_y_p2, _df_y_p3]
_df_y_p1_3_mean, _df_y_p1_3_std = _mean_std(_list_y_p1_3)

# Correct values for mean and std.dev. relative change: 1-2 days, FUTURE.
# Geometric mean.
_list_y_f1_2_geo = [_df_y_f1, _geo(_df_y_f2, n=2)]
_df_y_f1_2_geo_mean, _df_y_f1_2_geo_std = _mean_std(_list_y_f1_2_geo)

# Correct values for mean and std.dev. relative change: 1-2 days, PAST.
# Geometric mean.
_list_y_p1_2_geo = [_df_y_p1, _geo(_df_y_p2, n=2)]
_df_y_p1_2_geo_mean, _df_y_p1_2_geo_std = _mean_std(_list_y_p1_2_geo)

# Correct values for mean and std.dev. relative change: 1-3 days, FUTURE.
# Geometric mean.
_list_y_f1_3_geo = [_df_y_f1, _geo(_df_y_f2, n=2), _geo(_df_y_f3, n=3)]
_df_y_f1_3_geo_mean, _df_y_f1_3_geo_std = _mean_std(_list_y_f1_3_geo)

# Correct values for mean and std.dev. relative change: 1-3 days, PAST.
# Geometric mean.
_list_y_p1_3_geo = [_df_y_p1, _geo(_df_y_p2, n=2), _geo(_df_y_p3, n=3)]
_df_y_p1_3_geo_mean, _df_y_p1_3_geo_std = _mean_std(_list_y_p1_3_geo)

# Correct values for mean and std.dev. relative change: 1-2 days, FUTURE.
# Geometric mean and log-transform.
_list_y_f1_2_geo_log = [_geo_log(_df_y_f1), _geo_log(_df_y_f2, n=2)]
_df_y_f1_2_geo_log_mean, _df_y_f1_2_geo_log_std = \
    _mean_std(_list_y_f1_2_geo_log)

# Correct values for mean and std.dev. relative change: 1-2 days, PAST.
# Geometric mean and log-transform.
_list_y_p1_2_geo_log = [_geo_log(_df_y_p1), _geo_log(_df_y_p2, n=2)]
_df_y_p1_2_geo_log_mean, _df_y_p1_2_geo_log_std = \
    _mean_std(_list_y_p1_2_geo_log)

# Correct values for mean and std.dev. relative change: 1-3 days, FUTURE.
# Geometric mean and log-transform.
_list_y_f1_3_geo_log = [_geo_log(_df_y_f1), _geo_log(_df_y_f2, n=2),
                        _geo_log(_df_y_f3, n=3)]
_df_y_f1_3_geo_log_mean, _df_y_f1_3_geo_log_std = \
    _mean_std(_list_y_f1_3_geo_log)

# Correct values for mean and std.dev. relative change: 1-3 days, PAST.
# Geometric mean and log-transform.
_list_y_p1_3_geo_log = [_geo_log(_df_y_p1), _geo_log(_df_y_p2, n=2),
                        _geo_log(_df_y_p3, n=3)]
_df_y_p1_3_geo_log_mean, _df_y_p1_3_geo_log_std = \
    _mean_std(_list_y_p1_3_geo_log)

###############################################################################
# rel_change()

def test_rel_change_ser():
    """Test investops.rel_change.rel_change() with Pandas Series"""

    # Test 1
    ser_y = rel_change(df=_ser_x1, freq='d', future=True, days=1)
    assert_series_equal(ser_y, _ser_y1_f1)

    # Test 2
    ser_y = rel_change(df=_ser_x2, freq='d', future=True, days=1)
    assert_series_equal(ser_y, _ser_y2_f1)

    # Test 3
    ser_y = rel_change(df=_ser_x3, freq='d', future=True, days=1)
    assert_series_equal(ser_y, _ser_y3_f1)

    # Test 4
    ser_y = rel_change(df=_ser_x1, freq='d', future=False, days=1)
    assert_series_equal(ser_y, _ser_y1_p1)

    # Test 5
    ser_y = rel_change(df=_ser_x2, freq='d', future=False, days=1)
    assert_series_equal(ser_y, _ser_y2_p1)

    # Test 6
    ser_y = rel_change(df=_ser_x3, freq='d', future=False, days=1)
    assert_series_equal(ser_y, _ser_y3_p1)


def test_rel_change_df():
    """Test investops.rel_change.rel_change() with Pandas DataFrame"""

    # Test 1
    df_y = rel_change(df=_df_x1, freq='d', future=True, days=1)
    assert_frame_equal(df_y, _df_y1_f1)

    # Test 2
    df_y = rel_change(df=_df_x2, freq='d', future=True, days=1)
    assert_frame_equal(df_y, _df_y2_f1)

    # Test 3
    df_y = rel_change(df=_df_x3, freq='d', future=True, days=1)
    assert_frame_equal(df_y, _df_y3_f1)

    # Test 4
    df_y = rel_change(df=_df_x, freq='d', future=True, days=1)
    assert_frame_equal(df_y, _df_y_f1)

    # Test 5
    df_y = rel_change(df=_df_x, freq='d', future=True, days=2)
    assert_frame_equal(df_y, _df_y_f2)

    # Test 6
    df_y = rel_change(df=_df_x, freq='d', future=True, days=3)
    assert_frame_equal(df_y, _df_y_f3)

    # Test 7
    df_y = rel_change(df=_df_x, freq='d', future=False, days=1)
    assert_frame_equal(df_y, _df_y_p1)

    # Test 8
    df_y = rel_change(df=_df_x, freq='d', future=False, days=2)
    assert_frame_equal(df_y, _df_y_p2)

    # Test 9
    df_y = rel_change(df=_df_x, freq='d', future=False, days=3)
    assert_frame_equal(df_y, _df_y_p3)

###############################################################################
# mean_rel_change()

def test_mean_rel_change():
    """
    Test investops.rel_change.mean_rel_change()
    WITHOUT geometric mean and log-transform.
    """

    # Test 1
    df_y_mean, df_y_std = \
        mean_rel_change(df=_df_x, freq='d', future=True, annualized=None,
                        min_days=1, max_days=2, log=False)
    assert_frame_equal(df_y_mean, _df_y_f1_2_mean)
    assert_frame_equal(df_y_std, _df_y_f1_2_std)

    # Test 2
    df_y_mean, df_y_std = \
        mean_rel_change(df=_df_x, freq='d', future=False, annualized=None,
                        min_days=1, max_days=2, log=False)
    assert_frame_equal(df_y_mean, _df_y_p1_2_mean)
    assert_frame_equal(df_y_std, _df_y_p1_2_std)

    # Test 3
    df_y_mean, df_y_std = \
        mean_rel_change(df=_df_x, freq='d', future=True, annualized=None,
                        min_days=1, max_days=3, log=False)
    assert_frame_equal(df_y_mean, _df_y_f1_3_mean)
    assert_frame_equal(df_y_std, _df_y_f1_3_std)

    # Test 4
    df_y_mean, df_y_std = \
        mean_rel_change(df=_df_x, freq='d', future=False, annualized=None,
                        min_days=1, max_days=3, log=False)
    assert_frame_equal(df_y_mean, _df_y_p1_3_mean)
    assert_frame_equal(df_y_std, _df_y_p1_3_std)


def test_mean_rel_change_geo():
    """Test investops.rel_change.mean_rel_change() WITH geometric mean."""

    # Test 1
    df_y_mean, df_y_std = \
        mean_rel_change(df=_df_x, freq='d', future=True, annualized=False,
                        min_days=1, max_days=2, log=False)
    assert_frame_equal(df_y_mean, _df_y_f1_2_geo_mean)
    assert_frame_equal(df_y_std, _df_y_f1_2_geo_std)

    # Test 2
    df_y_mean, df_y_std = \
        mean_rel_change(df=_df_x, freq='d', future=False, annualized=False,
                        min_days=1, max_days=2, log=False)
    assert_frame_equal(df_y_mean, _df_y_p1_2_geo_mean)
    assert_frame_equal(df_y_std, _df_y_p1_2_geo_std)

    # Test 3
    df_y_mean, df_y_std = \
        mean_rel_change(df=_df_x, freq='d', future=True, annualized=False,
                        min_days=1, max_days=3, log=False)
    assert_frame_equal(df_y_mean, _df_y_f1_3_geo_mean)
    assert_frame_equal(df_y_std, _df_y_f1_3_geo_std)

    # Test 4
    df_y_mean, df_y_std = \
        mean_rel_change(df=_df_x, freq='d', future=False, annualized=False,
                        min_days=1, max_days=3, log=False)
    assert_frame_equal(df_y_mean, _df_y_p1_3_geo_mean)
    assert_frame_equal(df_y_std, _df_y_p1_3_geo_std)


def test_mean_rel_change_geo_log():
    """
    Test investops.rel_change.mean_rel_change()
    WITH both geometric mean and log-transform.
    """

    # Test 1
    df_y_mean, df_y_std = \
        mean_rel_change(df=_df_x, freq='d', future=True, annualized=False,
                        min_days=1, max_days=2, log=True)
    assert_frame_equal(df_y_mean, _df_y_f1_2_geo_log_mean)
    assert_frame_equal(df_y_std, _df_y_f1_2_geo_log_std)

    # Test 2
    df_y_mean, df_y_std = \
        mean_rel_change(df=_df_x, freq='d', future=False, annualized=False,
                        min_days=1, max_days=2, log=True)
    assert_frame_equal(df_y_mean, _df_y_p1_2_geo_log_mean)
    assert_frame_equal(df_y_std, _df_y_p1_2_geo_log_std)

    # Test 3
    df_y_mean, df_y_std = \
        mean_rel_change(df=_df_x, freq='d', future=True, annualized=False,
                        min_days=1, max_days=3, log=True)
    assert_frame_equal(df_y_mean, _df_y_f1_3_geo_log_mean)
    assert_frame_equal(df_y_std, _df_y_f1_3_geo_log_std)

    # Test 4
    df_y_mean, df_y_std = \
        mean_rel_change(df=_df_x, freq='d', future=False, annualized=False,
                        min_days=1, max_days=3, log=True)
    assert_frame_equal(df_y_mean, _df_y_p1_3_geo_log_mean)
    assert_frame_equal(df_y_std, _df_y_p1_3_geo_log_std)

###############################################################################
