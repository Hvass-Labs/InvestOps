###############################################################################
#
# Functions for calculating the relative changes over time for a time-series.
# This is used to calculate growth-rates for financial data such as Sales and
# Earnings Growth, as well as annualized stock-returns.
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
import pandas as pd
from numba import jit, prange

from investops.constants import (BDAYS_PER_YEAR, DAYS_PER_YEAR, WEEKS_PER_YEAR,
                                 MONTHS_PER_YEAR, QUARTERS_PER_YEAR)

###############################################################################

def convert_to_periods(freq, bdays=0, days=0, weeks=0,
                       months=0, quarters=0, years=0):
    """
    Convert the number of days, weeks, months, quarters and years into the
    equivalent number of periods or time-steps in a time-series with the given
    frequency. This is useful e.g. when calculating annualized returns.

    We do not use Pandas built-in support for time-series frequencies, because
    it is complicated and does not seem to work well for financial data where
    the frequency can be e.g. quarterly, but the time-stamps may be different.

    :param freq:
        String for the frequency of the data. Valid options:
        - 'bdays' or 'b' for business or trading-days data.
        - 'days' or 'd' for data that has all 7 week-days.
        - 'weeks' or 'w' for weekly data.
        - 'months' or 'm' for monthly data.
        - 'quarters' or 'q' for quarterly data.
        - 'ttm' for trailing-twelve-months data.
        - 'years', 'y', 'annual', 'a' for yearly or annual data.

    :param bdays: Number of business or trading-days.
    :param days: Number of days in a 7-day week.
    :param weeks: Number of weeks.
    :param months: Number of months.
    :param quarters: Number of quarters.
    :param years: Number of years.

    :return:
        - periods (int): Equivalent number of periods for the time-series.
        - total_years (float): Equivalent number of years for the time-series.
    """

    # First we calculate the total number of years from all the arguments.
    total_years = bdays / BDAYS_PER_YEAR \
                  + days / DAYS_PER_YEAR \
                  + weeks / WEEKS_PER_YEAR \
                  + months / MONTHS_PER_YEAR \
                  + quarters / QUARTERS_PER_YEAR \
                  + years

    # Then we will convert total_years into the equivalent number of periods
    # (or steps) for a time-series with the given frequency, and finally
    # re-calculate total_years to match the rounded number of integer periods.

    # Ensure the string with the freq argument is lower-case.
    freq = freq.lower()

    # DataFrame's frequency is Business or Trading Days.
    if freq in ['bdays', 'b']:
        # Rounded number of periods / time-steps.
        periods = round(BDAYS_PER_YEAR * total_years)
        # Re-calculate the total number of years using the rounded number.
        total_years = periods / BDAYS_PER_YEAR

    # DataFrame's frequency is Days (all 7 week-days).
    elif freq in ['days', 'd']:
        # Rounded number of periods / time-steps.
        periods = round(DAYS_PER_YEAR * total_years)
        # Re-calculate the total number of years using the rounded number.
        total_years = periods / DAYS_PER_YEAR

    # DataFrame's frequency is Weeks.
    elif freq in ['weeks', 'w']:
        # Rounded number of periods / time-steps.
        periods = round(WEEKS_PER_YEAR * total_years)
        # Re-calculate the total number of years using the rounded number.
        total_years = periods / WEEKS_PER_YEAR

    # DataFrame's frequency is Months.
    elif freq in ['months', 'm']:
        # Rounded number of periods / time-steps.
        periods = round(MONTHS_PER_YEAR * total_years)
        # Re-calculate the total number of years using the rounded number.
        total_years = periods / MONTHS_PER_YEAR

    # DataFrame's frequency is Quarters / TTM.
    elif freq in ['quarters', 'q', 'ttm']:
        # Rounded number of periods / time-steps.
        periods = round(QUARTERS_PER_YEAR * total_years)
        # Re-calculate the total number of years using the rounded number.
        total_years = periods / QUARTERS_PER_YEAR

    # DataFrame's frequency is Years.
    elif freq in ['years', 'y', 'annual', 'a']:
        # Rounded number of periods / time-steps.
        periods = round(total_years)
        # Re-calculate the total number of years using the rounded number.
        total_years = periods

    # Error.
    else:
        msg = f'Unsupported arg freq=\'{freq}\''
        raise ValueError(msg)

    # Convert periods to int, it should already be rounded to nearest number.
    periods = int(periods)

    return periods, total_years

###############################################################################
# Relative Change.

def rel_change(df, freq, future=True,
               bdays=0, days=0, weeks=0, months=0, quarters=0, years=0,
               annualized=False, fill_method=None, new_names=None):
    """
    Calculate the relative change for the values in a Pandas DataFrame or
    Series. This can be used to calculate e.g. Sales and Earnings Growth,
    as well as stock-returns.

    This is similar to Pandas built-in function `pct_change` but the interval
    is specified as a combination of days, weeks, months, quarters and years.
    And the relative changes can also be calculated as annualized numbers,
    which is useful for e.g. multi-year stock-returns.

    The number of days, weeks, months, quarters and years is combined into
    an integer `periods` for the number of time-steps to shift the time-series,
    depending on the time-series frequency. The relative change is calculated
    from the original and shifted time-series.

    If `annualized==False` then the function calculates the following:

    - If `future==True` then `df_result[i] = df[i+periods] / df[i] - 1`
    - If `future==False` then `df_result[i] = df[i] / df[i-periods] - 1`

    If `annualized==True` then the function calculates the annualized change
    instead, which is particularly useful when the time-interval is several
    years. For example, this can be used to calculate the Annualized Total
    Return on stocks. The variable `total_years` is the number of years
    corresponding to `periods`. So the function calculates:

    - If `future==True` then
      `df_result[i] = (df[i+periods] / df[i]) ** (1 / total_years) - 1`
    - If `future==False` then
      `df_result[i] = (df[i] / df[i-periods]) ** (1 / total_years) - 1`

    :param df:
        Pandas DataFrame or Series with time-series data, such as stock-prices
        or other financial data such as the annual Earnings Per Share. If this
        is a DataFrame then the rows are for the time-steps and the columns
        are for the different time-series that will be treated individually.

        .. warning:: `df` is assumed to be sorted in ascending order on its
            index. And the time-series data is assumed to be complete in the
            sense that data is present for all time-steps at the given
            frequency, otherwise you need to fill in the missing data before
            calling this function, or use the `fill_method` argument.

    :param freq:
        String for the frequency of the time-series data `df`. Valid options:

        - 'bdays' or 'b' for business or trading-days data.
        - 'days' or 'd' for data that has all 7 week-days.
        - 'weeks' or 'w' for weekly data.
        - 'months' or 'm' for monthly data.
        - 'quarters' or 'q' for quarterly data.
        - 'years', 'y', 'annual', 'a' for yearly or annual data.

    :param future:
        Boolean whether to calculate the future (True) or past (False) change.

    :param bdays: Number of business or trading-days.
    :param days: Number of days in a 7-day week.
    :param weeks: Number of weeks.
    :param months: Number of months.
    :param quarters: Number of quarters.
    :param years: Number of years.

    :param annualized:
        Boolean whether to calculate the annualized change (True)
        or the relative change (False). When calculating the change over
        several years, it is often useful to calculate the annualized change
        by setting this to True.

    :param fill_method:
        String for the method to fill in missing values in the input `df`.
        This is passed directly to the Pandas `fillna` method.

    :param new_names:
        If `df` is a Pandas Series, then this is a single string.
        If `df` is a DataFrame, then this is a dict with new column-names,
        or a mapper-function that will be passed to Pandas `rename` function.

    :return:
        Pandas DataFrame or Series.
    """
    # Check input data-type.
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        msg = 'Argument \'df\' must be either a Pandas DataFrame or Series.'
        raise TypeError(msg)

    # Fill in missing values in the data?
    if fill_method is not None:
        df = df.fillna(method=fill_method)

    # Convert the arguments to the equivalent number of periods (int) that
    # the time-series must be shifted, and the total number of years (float)
    # that it corresponds to, which is used in the annualized formula below.
    periods, total_years = \
        convert_to_periods(freq=freq, bdays=bdays, days=days, weeks=weeks,
                           months=months, quarters=quarters, years=years)

    if future:
        # Relative change between step [i] and future time-step [i+periods].
        # This calculates: df_result[i] = df[i+periods] / df[i]
        df_result = df.shift(periods=-periods) / df
    else:
        # Relative change between past time-step [i-periods] and step [i].
        # This calculates: df_result[i] = df[i] / df[i-periods]
        df_result = df / df.shift(periods=periods)

    if annualized:
        # Calculate the annualized change.
        df_result = df_result ** (1.0 / total_years) - 1.0
    else:
        # Finalize the relative change by subtracting 1.
        df_result = df_result - 1.0

    # Rename the Pandas Series, or rename the columns in the DataFrame.
    if new_names is not None:
        df_result.rename(new_names, axis=1, inplace=True)

    return df_result

###############################################################################
# Mean Relative Change.

@jit
def _calc_indices(future, num_steps, min_periods, max_periods):
    """
    Helper-function for determining the start / end-indices for iterating
    over the time-series, as well as the start / end-offsets for slicing.

    :param future:
        Boolean whether to calculate the future (True) or past (False) change.

    :param num_steps:
        Integer with the number of time-steps in the time-series data.

    :param min_periods:
        Integer with the min number of periods / time-steps for the windowing.

    :param max_periods:
        Integer with the max number of periods / time-steps for the windowing.

    :return:
        - start_idx: Int with the start-index for iterating over the data.
        - end_idx: Int with the end-index for iterating over the data.
        - start_offset: Int with the start-offset for slicing the data.
        - end_offset: Int with the end-offset for slicing the data.
    """
    if future:
        # Used for calculating the FUTURE mean relative changes.

        # For-loop start/end indices.
        start_idx = 0
        end_idx = num_steps - max_periods

        # Window-slicing start/end offsets.
        start_offset = min_periods
        end_offset = max_periods
    else:
        # Used for calculating the PAST mean relative changes.

        # For-loop start/end indices when calculating PAST relative changes.
        start_idx = max_periods
        end_idx = num_steps

        # Window-slicing start/end offsets.
        start_offset = -max_periods
        end_offset = -min_periods

    # Because Numpy slices are inclusive on the left end-point but exclusive
    # on the right end-point, we need to add 1 to the right end-point so both
    # the left and right end-points are included in the window / slice.
    end_offset += 1

    return start_idx, end_idx, start_offset, end_offset


@jit(parallel=False)
def _mean_log_change(values, exponent, future, min_periods, max_periods,
                     result_mean, result_std):
    """
    Helper-function for calculating the mean relative change using a
    log-transform of the time-series data, which is much faster because
    it avoids the need for making slow exponential / power computations.

    For relative changes in the time-series values between -20% and +20%,
    the log-transform gives results that are very close to the non-transformed
    data. But for relative changes beyond that range, the errors get
    increasingly bigger.

    This is a fast implementation of the algorithm using Numba JIT.

    :param values:
        Numpy 2-dim array for the input time-series values. The rows are
        for the time-steps and the columns are for the different time-series.

    :param exponent:
        Numpy 1-dim array with the exponent for each time-step in a slice,
        used to calculate the annualized change, or the geometric averages.
        If `None` then the relative changes are used directly.

    :param future:
        Boolean whether to calculate the future (True) or past (False) change.

    :param min_periods:
        Integer with the min number of periods / time-steps from the current
        time-step, when taking a slice to compute the mean relative change.

    :param max_periods:
        Integer with the max number of periods / time-steps from the current
        time-step, when taking a slice to compute the mean relative change.

    :param result_mean:
        Numpy 2-dim array for saving the calculated mean relative changes.
        Assumed to contain all `np.nan` values.

    :param result_std:
        Numpy 2-dim array for saving the calculated std.dev. relative changes.
        Assumed to contain all `np.nan` values.

    :return:
        None
    """
    # Calculate the for-loop indices and slicing offsets that we need below.
    start_idx, end_idx, start_offset, end_offset = \
        _calc_indices(future=future, num_steps=len(values),
                      min_periods=min_periods, max_periods=max_periods)

    # Log-transform of the time-series values.
    log_values = np.log(values)

    # For each time-series which is a column in the 2-dim array.
    for k in prange(values.shape[1]):
        # Note: There is no speed-up from taking a column from the 2-dim array
        # and working on it like this: column = log_values[:, k] as we might
        # imagine would helpe improve CPU-cache usage, but it doesn't.

        # For each time-step between the start/end indices, which depend on
        # whether we are calculating the future or past mean rel.changes.
        for i in range(start_idx, end_idx):
            # Time-series value for the current time-step.
            log_value_i = log_values[i, k]

            # Time-series values for a slice / window of time-steps.
            log_values_slice = log_values[i + start_offset:i + end_offset, k]

            # Relative log-changes between the current value and slice values.
            if future:
                # This is an array that is equivalent to calculating:
                # log_values[i+min_periods:i+max_periods+1]) / log_values[i]
                rel_log_changes = log_values_slice - log_value_i
            else:
                # This is an array that is equivalent to calculating:
                # log_values[i] / log_values[i-max_periods:i-min_periods+1])
                rel_log_changes = log_value_i - log_values_slice

            # Transform into either annualized changes or geometric averages.
            if exponent is not None:
                # This uses a mathematical property of the log-transform to
                # avoid the slow exponential calculation.
                rel_log_changes = exponent * rel_log_changes

            # Calculate and save the mean and std.dev. for the relative changes
            # of the current slice / window of the time-series.
            result_mean[i, k] = np.mean(rel_log_changes)
            result_std[i, k] = np.std(rel_log_changes)


# Parallel Numba JIT version of the function _mean_log_change.
_mean_log_change_par = jit(_mean_log_change.py_func, parallel=True)


@jit(parallel=False)
def _mean_rel_change(values, exponent, future, min_periods, max_periods,
                     result_mean, result_std):
    """
    Helper-function for calculating the mean relative change of time-series
    data. This uses the proper exponential calculation of e.g. annualized
    changes, which is the correct way but much slower than the log-transform.

    This is a fast implementation of the algorithm using Numba JIT.

    :param values:
        Numpy 2-dim array for the input time-series values. The rows are
        for the time-steps and the columns are for the different time-series.

    :param exponent:
        Numpy 1-dim array with the exponent for each time-step in a slice,
        used to calculate the annualized change, or the geometric averages.
        If `None` then the relative changes are used directly.

    :param future:
        Boolean whether to calculate the future (True) or past (False) change.

    :param min_periods:
        Integer with the min number of periods / time-steps from the current
        time-step, when taking a slice to compute the mean relative change.

    :param max_periods:
        Integer with the max number of periods / time-steps from the current
        time-step, when taking a slice to compute the mean relative change.

    :param result_mean:
        Numpy 2-dim array for saving the calculated mean relative changes.
        Assumed to contain all `np.nan` values.

    :param result_std:
        Numpy 2-dim array for saving the calculated std.dev. relative changes.
        Assumed to contain all `np.nan` values.

    :return:
        None
    """
    # Calculate the for-loop indices and slicing offsets that we need below.
    start_idx, end_idx, start_offset, end_offset = \
        _calc_indices(future=future, num_steps=len(values),
                      min_periods=min_periods, max_periods=max_periods)

    # For each time-series which is a column in the 2-dim array.
    for k in prange(values.shape[1]):
        # Note: There is no speed-up from taking a column from the 2-dim array
        # and working on it like this: column = values[:, k] as we might
        # imagine would help improve CPU-cache usage, but it doesn't.

        # For each time-step between the start/end indices, which depend on
        # whether we are calculating the future or past mean rel.changes.
        for i in range(start_idx, end_idx):
            # Time-series value for the current time-step.
            value_i = values[i, k]

            # Time-series values for a slice / window of time-steps.
            values_slice = values[i + start_offset:i + end_offset, k]

            # Relative changes between the current value and slice of values.
            if future:
                # This is an array that is equivalent to calculating:
                # values[i + min_periods:i + max_periods + 1]) / values[i]
                rel_changes = values_slice / value_i
            else:
                # This is an array that is equivalent to calculating:
                # values[i] / values[i - max_periods:i - min_periods + 1])
                rel_changes = value_i / values_slice

            # Transform into either annualized changes or geometric averages.
            if exponent is not None:
                # This uses the accurate but slow exponential calculation.
                rel_changes = rel_changes ** exponent - 1.0
            else:
                # Don't adjust for time. Just finalize the relative changes.
                rel_changes -= 1.0

            # Calculate and save the mean and std.dev. for the relative changes
            # of the current slice / window of the time-series.
            result_mean[i, k] = np.mean(rel_changes)
            result_std[i, k] = np.std(rel_changes)


# Parallel Numba JIT version of the function _mean_rel_change.
_mean_rel_change_par = jit(_mean_rel_change.py_func, parallel=True)


def mean_rel_change(df, freq, future=True, annualized=False, log=False,
                    parallel=True, fill_method=None,
                    min_bdays=0, min_days=0, min_weeks=0,
                    min_months=0, min_quarters=0, min_years=0,
                    max_bdays=0, max_days=0, max_weeks=0,
                    max_months=0, max_quarters=0, max_years=0,
                    new_names_mean=None, new_names_std=None):
    """
    Calculate the mean relative change in a time-series for a range of periods.

    This is useful e.g. for calculating the mean future stock-returns for all
    periods between e.g. 1 and 3 years to smoothen out short-term volatility.
    This can make it easier to see how predictor variables such as P/E ratios
    may relate to future returns.

    Instead of calculating the relative change between two points `df[i]` and
    `df[i + periods]` as done in the function `rel_change`, we want to calculate
    the mean change between a point `df[i]` and a whole slice of points
    `df[i + min_periods:i + max_periods]`. Furthermore, we may want to calculate
    the mean annualized changes, or the geometric mean changes to take the time
    into account.

    It is also possible to use a log-transform in this function, which gives
    nearly the same results when the relative changes are moderate, but is much
    faster to compute.

    :param df:
        Pandas DataFrame or Series with time-series data, such as stock-prices
        or other financial data such as the annual Earnings Per Share. If this
        is a DataFrame then the rows are for the time-steps and the columns
        are for the different time-series that will be treated individually.

        .. warning:: `df` is assumed to be sorted in ascending order on its
            index. And the time-series data is assumed to be complete in the
            sense that data is present for all time-steps at the given
            frequency, otherwise you need to fill in the missing data before
            calling this function, or use the `fill_method` argument.

    :param freq:
        String for the frequency of the time-series data `df`. Valid options:

        - 'bdays' or 'b' for business or trading-days data.
        - 'days' or 'd' for data that has all 7 week-days.
        - 'weeks' or 'w' for weekly data.
        - 'months' or 'm' for monthly data.
        - 'quarters' or 'q' for quarterly data.
        - 'years', 'y', 'annual', 'a' for yearly or annual data.

    :param future:
        Boolean whether to calculate the future (True) or past (False) change.

    :param annualized:
        Boolean whether to calculate the annualized change (True),
        or the geometric mean with the original frequency of the data (False).
        For example, if you want to calculate the change over several years,
        it is often useful to calculate the annualized change by setting this
        to True. But if you want to calculate the change over shorter periods
        e.g. days or weeks, then the annualized change may result in extreme
        values. If you set this to False then the geometric mean is calculated
        in the original frequency of the data, e.g. daily for share-price data
        or quarterly for quarterly financial data. You can also disable this
        time-based adjustment by setting this argument to `None`.

    :param log:
        Boolean whether to use a log-transform of the time-series data (True),
        or use the data as given (False). The log-transform is much faster to
        compute because it transforms the slow exponential computations into
        mathematically simpler and much faster operations. For relative changes
        in the time-series between -20% and +20%, the log-transform gives
        results that are very close to the non-transformed data. But for
        relative changes beyond that range, the errors get increasingly bigger.
        You should test on your own data if these errors are acceptable to you.
        The computation time is around 13x faster when using the log-transform.

    :param parallel:
        Boolean whether to compute in parallel (True) or serial (False).

    :param fill_method:
        String for the method to fill in missing values in the input `df`.
        This is passed directly to the Pandas `fillna` method.

    :param min_bdays: Min number of business or trading-days.
    :param min_days: Min number of days in a 7-day week.
    :param min_weeks: Min number of weeks.
    :param min_months: Min number of months.
    :param min_quarters: Min number of quarters.
    :param min_years: Min number of years.

    :param max_bdays: Max number of business or trading-days.
    :param max_days: Max number of days in a 7-day week.
    :param max_weeks: Max number of weeks.
    :param max_months: Max number of months.
    :param max_quarters: Max number of quarters.
    :param max_years: Max number of years.

    :param new_names_mean:
        Used to replace the original names for the MEAN relative changes.
        If `df` is a Pandas Series, then this is a single string.
        If `df` is a DataFrame, then this is a dict with new column-names,
        or a mapper-function that will be passed to Pandas `rename` function.

    :param new_names_std:
        Used to replace the original names for the STD.DEV. relative changes.
        If `df` is a Pandas Series, then this is a single string.
        If `df` is a DataFrame, then this is a dict with new column-names,
        or a mapper-function that will be passed to Pandas `rename` function.

    :return:
        - Pandas DataFrame or Series with the MEAN relative changes.
        - Pandas DataFrame or Series with the STD.DEV. relative changes.
    """
    # Check input data is either Pandas DataFrame or Series.
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        msg = 'Argument \'df\' must be either a Pandas DataFrame or Series.'
        raise TypeError(msg)

    # Fill in missing values in the data?
    if fill_method is not None:
        df = df.fillna(method=fill_method)

    # Convert input data.
    if isinstance(df, pd.DataFrame):
        # Convert Pandas DataFrame directly to a 2-dim Numpy array.
        values = df.to_numpy()
    elif isinstance(df, pd.Series):
        # Convert Pandas Series to a 2-dim Numpy array, by inserting a new
        # dimension so the data is in one column. This is because the functions
        # that implement the actual algorithms expect 2-dim Numpy arrays.
        values = df.to_numpy()[:, np.newaxis]

    ###########################################################################
    # Convert the arguments to the equivalent number of periods (int) that the
    # time-series data must be shifted, and the total number of years (float)
    # that it corresponds to, which is used in the annualized formulas below.

    # Convert arguments for min_periods.
    min_periods, min_total_years = \
        convert_to_periods(freq=freq, bdays=min_bdays, days=min_days,
                           weeks=min_weeks, months=min_months,
                           quarters=min_quarters, years=min_years)

    # Convert arguments for max_periods.
    max_periods, max_total_years = \
        convert_to_periods(freq=freq, bdays=max_bdays, days=max_days,
                           weeks=max_weeks, months=max_months,
                           quarters=max_quarters, years=max_years)

    # Check the periods are valid.
    if min_periods >= max_periods:
        msg = 'The combined max periods must exceed the combined min periods.'
        raise ValueError(msg)

    # Number of periods in a slice / window of time-series data.
    num_periods = max_periods - min_periods

    ###########################################################################
    # Create the array of exponents for the window-slice, which is used to
    # either calculate the annualized change, or the geometric mean change.

    if annualized is True:
        # Generate values between 0.0 and 1.0 for each time-step in the slice.
        # Note we use num_periods + 1 because we will include both end-points.
        x = np.arange(num_periods + 1) / (num_periods)

        # Equivalent number of years the time-series data is shifted,
        # for each time-step in a slice / window.
        total_years = (max_total_years - min_total_years) * x + min_total_years

        # Array of exponents used to calculate annualized change.
        exponent = 1.0 / total_years

    elif annualized is False:
        # Array of exponents used to calculate geometric mean change in the
        # same frequency as the original data, e.g. mean daily change for
        # share-price data, or mean quarterly change for quarterly data.
        # Note that we use max_periods + 1 to include the right end-point.
        exponent = 1.0 / np.arange(min_periods, max_periods + 1)

    else:
        # Disable time-based adjustment when calculating the relative changes.
        exponent = None

    # Reverse the array of exponents if we want to calculate the
    # relative changes from PAST time-steps to the current time-step.
    if exponent is not None and not future:
        exponent = np.flip(exponent)

    ###########################################################################
    # Main computation.

    # Pre-allocate Numpy arrays for the results.
    result_mean = np.full_like(values, fill_value=np.nan, dtype=float)
    result_std = np.full_like(values, fill_value=np.nan, dtype=float)

    # Common arguments for the functions below.
    args = dict(values=values, exponent=exponent, future=future,
                min_periods=min_periods, max_periods=max_periods,
                result_mean=result_mean, result_std=result_std)

    # Do the actual computation using a fast implementation with Numba JIT.
    # We need to switch between using log-transform or the raw data,
    # as well as switch between using parallel or serial execution.
    if log:
        if parallel:
            # Do computation using LOG-transform and PARALLEL execution.
            _mean_log_change_par(**args)
        else:
            # Do computation using LOG-transform and SERIAL execution.
            _mean_log_change(**args)
    else:
        if parallel:
            # Do computation using the RAW data and PARALLEL execution.
            _mean_rel_change_par(**args)
        else:
            # Do computation using the RAW data and SERIAL execution.
            _mean_rel_change(**args)

    ###########################################################################
    # Convert result to Pandas data.

    # Convert Numpy to Pandas DataFrame/Series similar to the input data.
    if isinstance(df, pd.Series):
        # Common arguments.
        args = dict(index=df.index, name=df.name)

        # Convert to Pandas Series. Only use the first column of the data.
        df_result_mean = pd.Series(data=result_mean[:, 0], **args)
        df_result_std = pd.Series(data=result_std[:, 0], **args)
    elif isinstance(df, pd.DataFrame):
        # Common arguments.
        args = dict(index=df.index, columns=df.columns)

        # Convert to Pandas DataFrame. Use all columns of data.
        df_result_mean = pd.DataFrame(data=result_mean, **args)
        df_result_std = pd.DataFrame(data=result_std, **args)

    # Rename the Pandas Series/DataFrame for the MEAN relative changes.
    if new_names_mean is not None:
        df_result_mean.rename(new_names_mean, axis=1, inplace=True)

    # Rename the Pandas Series/DataFrame for the STD.DEV. relative changes.
    if new_names_std is not None:
        df_result_std.rename(new_names_std, axis=1, inplace=True)

    return df_result_mean, df_result_std

###############################################################################
