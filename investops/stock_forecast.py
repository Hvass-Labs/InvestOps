###############################################################################
#
# Long-Term Stock Forecasting.
#
# This implements the mathematical model from the paper:
# - M.E.H. Pedersen, "Long-Term Stock Forecasting", 2020.
#   https://ssrn.com/abstract=3750775
#   https://github.com/Hvass-Labs/Finance-Papers
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
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from investops.stats import weighted_mean_std
from investops.metrics import r_squared
from investops.utils import dropna

###############################################################################

class StockForecast:
    """
    Mathematical model used to forecast long-term stock returns, from a range
    of possible future dividend yields, growth-rates, and valuation ratios.

    If we could precisely predict the future div.yield, growth, and val.ratio,
    then we could precisely predict the future stock-return. But we cannot
    predict these with complete accuracy, so instead we use a range of
    reasonable guesses for the future values, and let the model calculate
    the mean and standard deviation for the future stock-returns.

    This implements the mathematical model from the paper:
    - M.E.H. Pedersen, "Long-Term Stock Forecasting", 2020.
      https://ssrn.com/abstract=3750775
      https://github.com/Hvass-Labs/Finance-Papers
    """

    def __init__(self, div_yield, growth, val_ratio, years, rng,
                 dependent=True, prob=None, prob_div_yield=None,
                 prob_growth=None, prob_val_ratio=None, num_samples=10000):
        """
        :param div_yield:
            List or array of possible future dividend yields.
            For example, the list [0.02, 0.05, 0.08] means that the future
            dividend yield can be either 2%, 5%, or 8%.

        :param growth:
            List or array of possible future annual growth-rates.
            For example, the list [-0.1, 0.02, 0.15] means that the future
            growth-rate is either -10%, +2%, or +15%. These must match the
            valuation ratios, so if using the P/Sales ratio then these
            growth-rates must be for the Sales Per Share.

        :param val_ratio:
            List or array of possible future valuation ratios.
            For example, the list [0.8, 1.05, 1.3] means that the future
            valuation ratio is either 0.8, 1.05, or 1.3. These could be
            e.g. P/Sales, P/E or P/Book ratios.

        :param years:
            Number of years for the stock forecast. This can either be an
            integer or a float for a fractional number of years.

        :param rng:
            `Numpy.random.Generator` object from `np.random.default_rng()`

        :param dependent:
            Boolean whether the arguments `div_yield`, `growth` and `val_ratio`
            are statistically dependent (True) or independent (False). If they
            are dependent then all three lists must have the same length, as
            the elements of the lists will be sampled together. If they are
            independent then the three lists can have different lengths, as
            they will be sampled independently when forecasting the future.

        :param prob:
            Either `None` or a list or array of probabilities for selecting the
            elements of the arguments `div_yield`, `growth` and `val_ratio`.
            If `None` then all elements have equal probability of occurring.
            These probabilities are only used if the arg `dependent` is True.

        :param prob_div_yield:
            Either `None` or a list or array of probabilities for selecting the
            elements of the argument `div_yield` when forecasting the future.
            If `None` then all elements have equal probability of occurring.
            These probabilities are only used if the arg `dependent` is False.

        :param prob_growth:
            Either `None` or a list or array of probabilities for selecting the
            elements of the argument `growth` when forecasting the future.
            If `None` then all elements have equal probability of occurring.
            These probabilities are only used if the arg `dependent` is False.

        :param prob_val_ratio:
            Either `None` or a list or array of probabilities for selecting the
            elements of the argument `val_ratio` when forecasting the future.
            If `None` then all elements have equal probability of occurring.
            These probabilities are only used if the arg `dependent` is False.

        :param num_samples:
            Integer with number of samples for the Monte Carlo simulation used
            to calculate the model parameter `b` when calculating the std.dev.
            in case `dependent` is False.
        """
        # Copy args to self.
        self._rng = rng
        self._years = years

        # Convert possible future values to Numpy arrays.
        val_ratio = np.asarray(val_ratio)
        # Add +1 to these so we don't have to do that several times below.
        div_yield = np.asarray(div_yield) + 1.0
        growth = np.asarray(growth) + 1.0

        # Remove NaN (Not-a-Number) from arrays.
        val_ratio = dropna(val_ratio)
        div_yield = dropna(div_yield)
        growth = dropna(growth)

        # Check the arguments are valid.
        self._check_args(div_yield=div_yield, growth=growth,
                         val_ratio=val_ratio, dependent=dependent,
                         prob=prob, prob_div_yield=prob_div_yield,
                         prob_growth=prob_growth,
                         prob_val_ratio=prob_val_ratio)

        # Annualized val.ratio for investment periods of given number of years.
        val_ratio_ann = val_ratio ** (1.0 / years)

        # Are the possible future values statistically dependent?
        if dependent:
            # Calculate the parameters for the forecasting model when
            # the future div.yield, growth and val.ratios are DEPENDENT.

            # Combine the three arrays of possible future values.
            # These are already +1 from above, so we don't add that again here.
            x = div_yield * growth * val_ratio_ann

            # Use probabilities for the possible future values?
            if prob is None:
                # All future values have equal probability of occurring.

                # Calculate parameter `a` for the forecasting model.
                self.a = np.mean(x)

                # Calculate parameter `b` for the forecasting model.
                # This is the biased estimate of the std.dev. because that is
                # consistent with how the std.dev. is calculated below when
                # the probabilities are also given.
                self.b = np.std(x)
            else:
                # Ensure the probabilities are a Numpy array,
                # because the following function requires that.
                prob = np.asarray(prob)

                # Calculate the probability-weighted mean and std.dev.
                # and use those as the parameters of the forecasting model.
                self.a, self.b = weighted_mean_std(x=x, weights=prob)
        else:
            # Calculate the parameters for the forecasting model when
            # the future div.yield, growth and val.ratios are INDEPENDENT.

            # Calculate the parameter `a` for the forecasting model, using the
            # probabilities for the different arrays, if they were given.
            self.a = np.average(div_yield, weights=prob_div_yield) \
                   * np.average(growth, weights=prob_growth) \
                   * np.average(val_ratio_ann, weights=prob_val_ratio)

            # To calculate the parameter `b` for the forecasting model, we will
            # now do a Monte Carlo simulation / resampling from the arrays of
            # future values. These are sampled INDEPENDENTLY of each other.

            # Sample the future dividend yield. Assume it is already +1.
            div_yield_sample = rng.choice(div_yield, size=num_samples,
                                          replace=True, p=prob_div_yield)

            # Sample the future growth. Assume it is already +1.
            growth_sample = rng.choice(growth, size=num_samples,
                                       replace=True, p=prob_growth)

            # Sample the future ann.val.ratio at time of selling the stock.
            val_ratio_sample = rng.choice(val_ratio_ann, size=num_samples,
                                          replace=True, p=prob_val_ratio)

            # Combine the three samples.
            # These are already +1, so we don't add that again here.
            x = div_yield_sample * growth_sample * val_ratio_sample

            # Calculate the parameter `b` for the forecasting model.
            self.b = np.std(x)

    @staticmethod
    def _check_args(div_yield, growth, val_ratio, dependent,
                    prob, prob_div_yield, prob_growth, prob_val_ratio):
        """Check the class-arguments are valid."""

        # Check dim of div_yield array, which should already be a Numpy array.
        if len(div_yield.shape) != 1:
            msg = 'Argument \'div_yield\' must be 1-dim list or array.'
            raise TypeError(msg)

        # Check dim of growth array, which should already be a Numpy array.
        if len(growth.shape) != 1:
            msg = 'Argument \'growth\' must be a 1-dim list or array.'
            raise TypeError(msg)

        # Check dim of val_ratio array, which should already be a Numpy array.
        if len(val_ratio.shape) != 1:
            msg = 'Argument \'val_ratio\' must be a 1-dim list or array.'
            raise TypeError(msg)

        if dependent:
            # Check length of arrays are all equal.
            if not (len(div_yield) == len(growth) == len(val_ratio)):
                msg = 'Arrays \'div_yield\', \'growth\', and \'val_ratio\' ' \
                      'must all have the same length.'
                raise ValueError(msg)

            # Check length of prob-array is correct.
            if prob is not None and len(prob) != len(div_yield):
                msg = 'Array \'prob\' must have same length as other arrays.'
                raise ValueError(msg)
        else:
            # Common error-message.
            msg = 'Arrays \'{0}\' and \'prob_{0}\' must have same length.'

            # Check prob_div_yield is correct length.
            if prob_div_yield is not None \
                    and len(prob_div_yield) != len(div_yield):
                raise ValueError(msg.format('div_yield'))

            # Check prob_growth is correct length.
            if prob_growth is not None and len(prob_growth) != len(growth):
                raise ValueError(msg.format('growth'))

            # Check prob_val_ratio is correct length.
            if prob_val_ratio is not None \
                    and len(prob_val_ratio) != len(val_ratio):
                raise ValueError(msg.format('val_ratio'))

    def forecast(self, cur_val_ratio):
        """
        Use the model to forecast the future stock-returns given the current
        valuation ratios. This can also be used with historical valuation
        ratios, to see how the forecasting model performed on historical data.

        :param cur_val_ratio:
            Float or list/array of floats with current valuation ratios.

        :return:
            - Mean annualized return. Either float or Numpy array.
            - Std.dev. for the ann. returns. Either float or Numpy array.
        """
        # Convert to Numpy array.
        cur_val_ratio = np.asarray(cur_val_ratio)

        # Annualized valuation ratios used in both formulas.
        cur_val_ratio_ann = cur_val_ratio ** (1 / self._years)

        # Forecast the mean and std.dev. for the annualized returns,
        # using Eq.(14) and Eq.(18) from the paper referenced above.
        mean = self.a / cur_val_ratio_ann - 1.0
        std = self.b / cur_val_ratio_ann

        return mean, std

    def random(self, min_val_ratio, max_val_ratio, num_samples=1000):
        """
        Generate random historical valuation ratios and ann. returns,
        which is useful e.g. for plotting with synthetic historical data.
        Note: Real-world stock-data looks very different from this random data.

        First the valuation ratios are generated randomly from a uniform
        distribution between the given min and max valuation ratios. Then
        the forecasting model is used to calculate the mean and std.dev.
        for those valuation ratios. Finally, those mean and std.dev. are
        used to generate random normal-distributed annualized returns.

        :param min_val_ratio: Float with min valuation ratio.
        :param max_val_ratio: Float with max valuation ratio.
        :param num_samples: Integer with number of random samples.
        :return:
            - Numpy array with random valuation ratios.
            - Numpy array with random annualized returns.
        """
        # Random uniform-distributed valuation ratios.
        val_ratios = \
            self._rng.uniform(min_val_ratio, max_val_ratio, size=num_samples)

        # Sort the random valuation ratios.
        val_ratios = np.sort(val_ratios)

        # Calculate the forecast mean and std.dev. for the random val. ratios.
        forecast_mean, forecast_std = self.forecast(cur_val_ratio=val_ratios)

        # Random normal-distributed annualized returns.
        ann_rets = self._rng.normal(loc=forecast_mean, scale=forecast_std)

        return val_ratios, ann_rets

    def r_squared(self, hist_val_ratios, hist_ann_rets):
        """
        Calculate the Coefficient of Determination R^2 for measuring the
        Goodness of Fit between the forecasted mean and actual ann. returns.

        An R^2 value of one means there is a perfect fit and the forecasting
        model explains all the variance in the data. An R^2 value of zero
        means the forecasting model does not explain any variance in the data.

        Note: Because the forecasting model is non-linear, the R^2 can become
        negative if the model fits poorly on data with a large variance.

        :param hist_val_ratios:
            Numpy array with historical valuation ratios.

        :param hist_ann_rets:
            Numpy array with historical annualized returns.

        :return:
            Float with the R^2 value.
        """
        # Calculate the forecast mean for the historical valuation ratios.
        forecast_mean, _ = self.forecast(cur_val_ratio=hist_val_ratios)

        # Calculate R^2 between the historical returns and forecast mean.
        return r_squared(y_true=hist_ann_rets, y_pred=forecast_mean)

    def make_title(self, ticker=None, min_years=None, max_years=None,
                   start_year=None, end_year=None):
        """
        Make a standardized title for the plot.

        :param ticker:
            Either `None` or string with the stock-ticker.

        :param min_years:
            Either `None` or integer with the min number of years used when
            calculating the mean ann. returns for the historical stock-prices.
            If 'min_years' and `max_years` are both `None` then use the
            original argument `years` from when the class was instantiated.

        :param max_years:
            Either `None` or integer with the max number of years used when
            calculating the mean ann. returns for the historical stock-prices.
            If 'min_years' and `max_years` are both `None` then use the
            original argument `years` from when the class was instantiated.

        :param start_year:
            Integer with the start-year for the historical data.

        :param end_year:
            Integer with the end-year for the historical data.

        :return:
            String with the plot-title.
        """
        # First part of the title for the stock-ticker.
        if ticker is not None:
            title1 = f'[{ticker}] '
        else:
            title1 = ''

        # Second part of the title for the years of the ann. returns.
        if min_years is not None and max_years is not None:
            title2 = f'{min_years}-{max_years} Year Mean Ann. Return'
        else:
            title2 = f'{self._years}-Year Ann. Return'

        # Third part of the title for the historical data-period.
        if start_year is not None and end_year is not None:
            title3 = f' ({start_year}-{end_year})'
        else:
            title3 = ''

        # Combine all titles.
        title = title1 + title2 + title3

        return title

    def plot(self, title, hist_val_ratios=None, hist_ann_rets=None,
             min_val_ratio=None, max_val_ratio=None, cur_val_ratio=None,
             name_val_ratio='Valuation Ratio', figsize=(10, 12)):
        """
        Create a plot of the forecasting model, showing how valuation ratios
        forecast the future mean and std.dev. for the annualized stock-returns.

        If historical data for the valuation ratios and ann.returns are given,
        then these are also shown as a scatter-plot, to see how well they fit
        the forecasting model. The dots are colored according to their time,
        so historical data-points that were close in time have similar colors.

        The R^2 shows the Goodness of Fit between the forecasted mean and the
        historical data. Because the forecasting model is non-linear, the R^2
        can be negative if the model fits poorly on data with high variance.

        Note: Even if the forecasting model has a very good fit on historical
        data, it does not mean that it will be accurate in the future. That
        depends entirely on how accurate your guesses are for the possible
        future dividend yields, growth-rates, and valuation ratios!

        :param title:
            String with the first part of the plot's title. You can use
            the class-method `make_title` to make a standardized title.

        :param hist_val_ratios:
            Either `None` or list / Numpy array / Pandas Series of
            historical valuation ratios.

        :param hist_ann_rets:
            Either `None` or list / Numpy array / Pandas Series of
            historical annualized returns.

        :param min_val_ratio:
            Float with the min valuation ratio for the x-axis.
            If `hist_val_ratios` is given then use the overall min.

        :param max_val_ratio:
            Float with the max valuation ratio for the x-axis.
            If `hist_val_ratios` is given then use the overall max.

        :param cur_val_ratio:
            Float with the current valuation ratio which will be shown
            as a vertical line on the plot.

        :param name_val_ratio:
            String with the name of the valuation ratio.

        :param figsize:
            Tuple with the figure-size which is passed to Matplotlib.

        :return:
            Matplotlib Figure object.
        """
        # Boolean whether to use historical data-points.
        use_hist = (hist_val_ratios is not None and hist_ann_rets is not None)

        if use_hist:
            # If historical data are Pandas Series then align the data index.
            if isinstance(hist_val_ratios, pd.Series) \
                    and isinstance(hist_ann_rets, pd.Series):
                # Combine the historical data into columns of one DataFrame.
                df_hist = pd.DataFrame(dict(val_ratios=hist_val_ratios,
                                            ann_rets=hist_ann_rets))

                # Remove rows with NaN (Not-a-Number).
                df_hist = df_hist.dropna()

                # Get the historical val.ratios.
                hist_val_ratios = df_hist['val_ratios'].to_numpy()

                # Get the historical ann.returns.
                hist_ann_rets = df_hist['ann_rets'].to_numpy()

            # Min valuation ratios for the x-axis.
            if min_val_ratio is None:
                min_val_ratio = np.min(hist_val_ratios)
            else:
                min_val_ratio = min(min_val_ratio, np.min(hist_val_ratios))

            # Max valuation ratios for the x-axis.
            if max_val_ratio is None:
                max_val_ratio = np.max(hist_val_ratios)
            else:
                max_val_ratio = max(max_val_ratio, np.max(hist_val_ratios))

        # Extend min/max val-ratio if necessary to hold the current val-ratio.
        if cur_val_ratio < min_val_ratio:
            min_val_ratio = cur_val_ratio
        elif cur_val_ratio > max_val_ratio:
            max_val_ratio = cur_val_ratio

        # Evenly spaced valuation ratios between min and max.
        cur_val_ratios = \
            np.linspace(start=min_val_ratio, stop=max_val_ratio, num=100)

        # Use the model to forecast the mean and std.dev. ann.returns
        # from the evenly spaced valuation ratios on the x-axis.
        forecast_mean, forecast_std = \
            self.forecast(cur_val_ratio=cur_val_ratios)

        # Create a new plot.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(211)

        # Plot the forecast mean.
        if use_hist:
            # Label also shows the R^2 between forecast and historical data.
            r2 = self.r_squared(hist_val_ratios=hist_val_ratios,
                                hist_ann_rets=hist_ann_rets)
            label = f'Forecast Mean (R^2 = {r2:.2f})'
        else:
            label = 'Forecast Mean'
        ax.plot(cur_val_ratios, forecast_mean, color='black', label=label)

        # Plot one standard deviation around the forecast mean.
        color = 'green'
        alpha = 0.3
        # Plot lines below and above mean.
        ax.plot(cur_val_ratios, forecast_mean - forecast_std, color=color,
                label='Forecast Mean $\pm$ 1 Std.Dev.')
        ax.plot(cur_val_ratios, forecast_mean + forecast_std, color=color)
        # Fill the areas.
        ax.fill_between(cur_val_ratios,
                        forecast_mean + forecast_std,
                        forecast_mean - forecast_std,
                        color=color, edgecolor=color, alpha=alpha)

        # Plot two standard deviations around the forecast mean.
        color = 'red'
        alpha = 0.1
        # Plot lines below and above mean.
        ax.plot(cur_val_ratios, forecast_mean - 2 * forecast_std, color=color,
                label='Forecast Mean $\pm$ 2 Std.Dev.')
        ax.plot(cur_val_ratios, forecast_mean + 2 * forecast_std, color=color)
        # Fill the areas.
        ax.fill_between(cur_val_ratios,
                        forecast_mean - forecast_std,
                        forecast_mean - 2 * forecast_std,
                        color=color, edgecolor=color, alpha=alpha)
        ax.fill_between(cur_val_ratios,
                        forecast_mean + forecast_std,
                        forecast_mean + 2 * forecast_std,
                        color=color, edgecolor=color, alpha=alpha)

        # Plot the historical data?
        if use_hist:
            # Scatter-plot with the historical val.ratios and ann.returns.
            # Each dot is colored according to its date (array-position).
            # The dots are rasterized (turned into pixels) to save space
            # when saving to vectorized graphics-file.
            n = len(hist_val_ratios)
            c = np.arange(n) / n
            label = 'Historical Returns'
            ax.scatter(hist_val_ratios, hist_ann_rets, marker='o', c=c,
                       cmap='plasma', label=label, rasterized=True)

            # Plot mean of historical ann. returns as horizontal line.
            hist_ann_ret_mean = np.mean(hist_ann_rets)
            label = f'Mean Hist. Return = {hist_ann_ret_mean:.1%}'
            ax.axhline(y=hist_ann_ret_mean, color='black',
                       linestyle='dashed', label=label)

        # Plot horizontal line for Ann. Return = 0.0%
        ax.axhline(y=0.0, color='black', linestyle='dotted')

        # Plot vertical line for Current Valuation Ratio.
        if cur_val_ratio is not None:
            label = f'Current {name_val_ratio} = {cur_val_ratio:.2f}'
            ax.axvline(x=cur_val_ratio, color='blue', linestyle='dashed',
                       label=label)

        # Show the labels for what we have just plotted.
        ax.legend(loc='upper right', framealpha=1.0)

        # Create plot-title.
        # First part of the title is supplied by the user.
        # Second part of the title is the formula for mean ann. return.
        title_mean = \
            f'E[Ann Return] = {self.a:.2f} / ({name_val_ratio} ^ (1/{self._years})) - 1'
        # Third part of the title is the formula for std.dev. ann. return.
        title_std = \
            f'Std[Ann Return] = {self.b:.2f} / ({name_val_ratio} ^ (1/{self._years}))'
        # Combine and set the plot-title.
        title_full = '\n'.join([title, title_mean, title_std])
        ax.set_title(title_full)

        # Convert y-ticks to percentages.
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

        # Set axes labels.
        ax.set_xlabel(name_val_ratio)
        ax.set_ylabel('Annualized Return')

        # Show grid.
        ax.grid()

        return fig

###############################################################################
