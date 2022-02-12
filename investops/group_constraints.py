###############################################################################
#
# Group constraints for an investment portfolio.
#
# This implements the algorithm from the paper:
# - M.E.H. Pedersen, "Portfolio Group Constraints", 2022.
#   https://ssrn.com/abstract=4033243
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
import numba as nb
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from investops.check import check_weights

###############################################################################
# Fast implementation of the Portfolio Group Constraint algorithms.

@jit
def _group_sums(weights, num_groups, asset_to_groups_nb):
    """
    Calculate the group-sums for the given portfolio weights.

    :param weights:
        Numpy array with portfolio weights.
        These can be both positive and negative combined.

    :param num_groups:
        Integer with the number of groups.

    :param asset_to_groups_nb:
        Numba list of Numpy arrays with group-id's.
        This is an efficient map from asset-id to a list of group-id's.

    :return:
        - Numpy array with group-sums for POSITIVE weights.
        - Numpy array with group-sums for NEGATIVE weights.
    """
    # Initialize the array of group-sums to zero.
    group_sums_pos = np.zeros(num_groups, dtype=np.float64)
    # For the negative group-sums, initialize the array with -0.0
    # so the sign is correct if e.g. used with the negative group-limits.
    group_sums_neg = -np.zeros(num_groups, dtype=np.float64)

    # For each asset-id and associated array of group-id's.
    # The algorithm described in the paper referenced above, iterates over
    # the groups instead of the assets, but the result is the exact same.
    for asset_id, array_group_id in enumerate(asset_to_groups_nb):
        # Portfolio weight for the given asset.
        w = weights[asset_id]

        # Add the portfolio weight for this asset to the group-sums
        # for all groups that are associated with this asset.
        if w >= 0.0:
            # Weight is positive (or zero), so add to positive group-sums.
            group_sums_pos[array_group_id] += w
        else:
            # Weight is negative, so add to negative group-sums.
            group_sums_neg[array_group_id] += w

    return group_sums_pos, group_sums_neg


@jit
def _group_ratios_signed(num_groups, group_sums, group_lim):
    """
    Calculate the ratios between the group-limits and the group-sums.
    These should be EITHER positive OR negative - NOT BOTH combined!

    A group-ratio will be `np.inf` (infinity) if the group-limit was inf
    or the group-sum was zero.

    :param num_groups:
        Integer with the number of groups.

    :param group_sums:
        Numpy array with the group-sums, with the same sign as `group_lim`.

    :param group_lim:
        Numpy array with the group-limits, with the same sign as `group_sums`.

    :return:
        Numpy array with the group-ratios.
    """
    # Initialize array with the group-ratios.
    group_ratios = np.empty(num_groups, dtype=np.float64)

    # For each group.
    for g in range(num_groups):
        # Avoid division by zero if the group-sum is zero,
        # as this would raise an exception.
        if group_sums[g] != 0.0:
            # Calculate the ratio between the group-limit and group-sum.
            ratio = group_lim[g] / group_sums[g]
        else:
            # The group-sum was zero so the group-ratio is infinity.
            ratio = np.inf

        # Save the ratio for this group.
        group_ratios[g] = ratio

    return group_ratios


@jit
def _group_ratios(num_groups, group_sums_pos, group_sums_neg,
                  group_lim_pos, group_lim_neg):
    """
    Calculate the ratios between the group-limits and the group-sums,
    for both positive and negative portfolio weights.

    A ratio below 1 means the group-sum is too high so the portfolio weights
    belonging to that group must be decreased by multiplying with that ratio.
    And a ratio above 1 means the group-sum is below its limit, so it would
    be possible to increase the portfolio weights belonging to that group by
    multiplying with that ratio, without exceeding the limit for that
    particular group (although it may then exceed the limit for other groups,
    which is handled in the function `_constrain_weights`).

    A group-ratio will be `np.inf` (infinity) if the group-limit was inf
    or the group-sum was zero.

    :param num_groups:
        Integer with the number of groups.

    :param group_sums_pos:
        Numpy array with the group-sums for POSITIVE portfolio weights.

    :param group_sums_neg:
        Numpy array with the group-sums for NEGATIVE portfolio weights.

    :param group_lim_pos:
        Numpy array with the group-limits for POSITIVE portfolio weights.

    :param group_lim_neg:
        Numpy array with the group-limits for NEGATIVE portfolio weights.

    :return:
        - Numpy array with group-ratios for POSITIVE portfolio weights.
        - Numpy array with group-ratios for NEGATIVE portfolio weights.
    """
    # Calculate the POSITIVE group-ratios.
    if group_lim_pos is not None:
        group_ratios_pos = _group_ratios_signed(num_groups=num_groups,
                                                group_sums=group_sums_pos,
                                                group_lim=group_lim_pos)
    else:
        # The POSITIVE group-limits are not being used.
        group_ratios_pos = None

    # Calculate the NEGATIVE group-ratios.
    if group_lim_neg is not None:
        group_ratios_neg = _group_ratios_signed(num_groups=num_groups,
                                                group_sums=group_sums_neg,
                                                group_lim=group_lim_neg)
    else:
        # The NEGATIVE group-limits are not being used.
        group_ratios_neg = None

    return group_ratios_pos, group_ratios_neg


@jit
def _constrain_weights(weights_org, num_groups, group_lim_pos, group_lim_neg,
                       asset_to_groups_nb, log=False, max_iter=100, tol=1e-6):
    """
    Adjust the portfolio weights so all the group-constraints are satisfied.
    This runs very fast using Numba Jit.

    This implements the main algorithm from the paper referenced above.

    :param weights_org:
        Numpy array with portfolio weights to be adjusted.

    :param num_groups:
        Integer with the number of groups.

    :param group_lim_pos:
        Numpy array with the group-limits for POSITIVE portfolio weights.

    :param group_lim_neg:
        Numpy array with the group-limits for NEGATIVE portfolio weights.

    :param asset_to_groups_nb:
        Numba list of Numpy arrays with group-id's.
        This is an efficient map from asset-id to a list of group-id's.

    :param log:
        Boolean whether to return all iterations of the weights (True),
        or just the final weights (False).

    :param max_iter:
        Integer with the max number of iterations for the algorithm.

    :param tol:
        Float with the convergence tolerance for aborting the algorithm,
        when all the weight-adjustments are below this number.
        This should be set to a small positive number such as 1e-6.

    :return:
        Numpy 2-dim array with the adjusted portfolio weights.
        If `log==True` the 2-dim array has the weights of all iterations.
        If `log==False` only the first row has the final adjusted weights.
    """
    # Copy the original portfolio weights as starting point for new weights.
    weights_new = weights_org.copy()

    # Number of assets.
    num_assets= len(weights_new)

    # Initialize the set / list of assets that remain to be processed.
    # The set data-structure allows us to remove an element in time O(1).
    remaining_assets = set(range(num_assets))

    # Calculate the group-sums for all groups with the initial weights.
    group_sums_pos, group_sums_neg = \
        _group_sums(weights=weights_new, num_groups=num_groups,
                    asset_to_groups_nb=asset_to_groups_nb)

    # Use a log for all iterations of the adjusted weights?
    if log:
        # Initialize an empty Numpy 2-dim array for logging the weights.
        # Normally we would use a list of 1-dim Numpy arrays instead,
        # but this creates some hassle with the return-types and Numba.
        # Note: This wastes memory if max_iter is much higher than needed.
        _log = np.zeros(shape=(max_iter+1, num_assets),
                        dtype=weights_new.dtype)

        # Log the initial weights to the first row of the 2-dim array.
        _log[0] = weights_new
    else:
        # The log is not used.
        _log = None

    # Repeat for a number of iterations, or until all weights have converged
    # so the loop can be aborted further below.
    for k in range(1, max_iter+1):
        # Calculate both the positive and negative group-ratios.
        group_ratios_pos, group_ratios_neg = \
            _group_ratios(num_groups=num_groups,
                          group_sums_pos=group_sums_pos,
                          group_sums_neg=group_sums_neg,
                          group_lim_pos=group_lim_pos,
                          group_lim_neg=group_lim_neg)

        # For each remaining asset whose weight still needs to be adjusted.
        # Note: The set is copied to avoid problems when removing elements
        # from it while iterating over it simultaneously.
        for asset_id in remaining_assets.copy():
            # Get the adjusted portfolio weight from the previous iteration.
            w_new = weights_new[asset_id]

            # If the adjusted portfolio weight is non-zero.
            if w_new != 0.0:
                # Originally desired portfolio weight.
                w_org = weights_org[asset_id]

                # Initialize the minimum ratio with the weight-ratio between
                # the original and adjusted portfolio weights. A ratio above 1
                # means the adjusted weight is smaller than the original
                # weight, so it is possible to increase it by this factor.
                # A ratio below 1 means that the portfolio weight is too large
                # and must be decreased (although this should never occur).
                min_ratio = weights_org[asset_id] / weights_new[asset_id]

                # Select the group-ratios depending on sign of weight.
                if w_new >= 0.0:
                    # Use the positive group-ratios.
                    group_ratios = group_ratios_pos
                else:
                    # Use the negative group-ratios.
                    group_ratios = group_ratios_neg

                if group_ratios is not None:
                    # Find the minimum of the weight-ratio and all group-ratios.
                    for group_id in asset_to_groups_nb[asset_id]:
                        if group_ratios[group_id] < min_ratio:
                            min_ratio = group_ratios[group_id]

                # If min_ratio is now below 1 it means that either the
                # adjusted portfolio weight is too large, or one or more of
                # the group-sums associated with this asset is too large,
                # so the portfolio weight must be decreased by this factor.
                # Conversely, if the min_ratio is now above 1, then it
                # means the adjusted portfolio weight is below the original
                # portfolio weight AND all the group-sums are below their
                # limits, so the weight can be increased by this factor.

                # The portfolio weight before it was adjusted.
                w_old = w_new

                # Adjust the portfolio weight.
                w_new = w_new * min_ratio

                # Clip the adjusted portfolio weight to ensure it is
                # less-or-equal to the original portfolio weight. This is
                # necessary due to the possibility of tiny floating-point
                # rounding errors. If float-math was completely accurate,
                # then this would not be necessary.
                if np.abs(w_new) > np.abs(w_org):
                    w_new = w_org

                # Save the adjusted portfolio weight.
                weights_new[asset_id] = w_new

                # Weight difference between current and previous iteration.
                w_dif = w_old - w_new

                # Get the list of group-id's associated with this asset.
                group_ids = asset_to_groups_nb[asset_id]

                # Update all relevant group-sums with the weight difference.
                if w_new >= 0.0:
                    # Update all the relevant group-sums for POSITIVE weights.
                    group_sums_pos[group_ids] -= w_dif
                else:
                    # Update all the relevant group-sums for NEGATIVE weights.
                    group_sums_neg[group_ids] -= w_dif

                # If the weight-adjustment was below tolerance, then this
                # weight has converged, so it can be removed from the list of
                # assets that remain to be processed.
                if np.abs(w_new - w_old) < tol:
                    remaining_assets.remove(asset_id)

        # Log the updated weights?
        if _log is not None:
            _log[k] = weights_new

        # If there are no assets remaining to be processed, abort the loop.
        if len(remaining_assets) == 0:
            break

    if _log is not None:
        # Only return the rows of the log that have actually been used.
        return _log[0:k+1]
    else:
        # Return only the final updated weights. Because Numba requires that
        # the different possible return-values have the same type and shape,
        # we must convert the 1-dim Numpy array to a 2-dim Numpy array.
        return weights_new.reshape((1, num_assets))

###############################################################################
# Main class.

class GroupConstraints:
    """
    Enforce group constraints for an investment portfolio, by lowering the
    portfolio weights until all group-sums are within their group-limits.

    This allows you to have many overlapping groups, for example, you could
    have overlapping groups for different asset-classes, countries, industries,
    credit-ratings, etc. And each asset can belong to any of these groups.

    For efficiency, you first setup the problem-solver by creating an instance
    of this class, which builds internal data-structures for fast execution
    using the Numba Jit compiler. This means that if you change the assets or
    the groups, then you MUST create a new instance of this class to rebuild
    the internal data-structures.

    When passing arguments to this class and its functions, all assets and
    groups must be named with unique identifiers such as stock-tickers for
    the assets and industry-names for the groups. This is to ensure that the
    portfolio weights, group-limits, etc. are properly identified.

    But internally the data gets converted to Numpy arrays and special Numba
    lists where the unique identifiers are replaced by integer id's, to speed
    up the computation. The returned data is converted back into Pandas Series
    with the correct asset-names and group-names for proper identification.

    This implements the algorithm from the paper:
    - M.E.H. Pedersen, "Portfolio Group Constraints", 2022.
      https://ssrn.com/abstract=4033243
      https://github.com/Hvass-Labs/Finance-Papers
    """
    def __init__(self, asset_names, group_names,
                 asset_to_groups, group_lim_pos, group_lim_neg):
        """
        :param asset_names:
            List of strings or other unique id's for the assets.
            You cannot use other asset-names later, you would then have to
            create a new `GroupConstraints` instance.

        :param group_names:
            List of strings or other unique id's for the groups.
            You cannot use other group-names later, you would then have to
            create a new `GroupConstraints` instance.

        :param asset_to_groups:
            Dict mapping from asset-names to lists of group-names.
            This defines the association between assets and groups.

        :param group_lim_pos:
            Pandas Series with the POSITIVE group-limits. Some of these limits
            can be set to `np.inf` which means those groups have no limits.
            The argument can also be set to `None` so there are no positive
            group-limits.

        :param group_lim_neg:
            Pandas Series with the NEGATIVE group-limits. Some of these limits
            can be set to `np.inf` which means those groups have no limits.
            The argument can also be set to `None` so there are no negative
            group-limits.
        """
        # Check the arguments are valid.
        # This hack re-uses all the args so we don't have to type them again.
        args = locals()
        args.pop('self')
        self._check_args(**args)

        # Convert to sorted lists of unique names.
        self._asset_names = sorted(set(asset_names))
        self._group_names = sorted(set(group_names))

        # Number of unique assets and groups.
        self._num_assets = len(self._asset_names)
        self._num_groups = len(self._group_names)

        # Positive group limits.
        if group_lim_pos is not None:
            # Convert to Numpy array ordered the same as self._group_names.
            self._group_lim_pos = group_lim_pos[self._group_names].to_numpy()
        else:
            # The positive group-limits are not used.
            self._group_lim_pos = None

        # Negative group limits.
        if group_lim_neg is not None:
            # Convert to Numpy array ordered the same as self._group_names.
            self._group_lim_neg = group_lim_neg[self._group_names].to_numpy()
        else:
            # The negative group-limits are not used.
            self._group_lim_neg = None

        # Create efficient map from asset names to their integer id's.
        self._asset_name_to_id = \
            {name: index for index, name in enumerate(self._asset_names)}

        # Create efficient map from group names to their integer id's.
        self._group_name_to_id = \
            {name: index for index, name in enumerate(self._group_names)}

        # Initialize empty maps from asset-id's to sets of group-id's.
        # This allows us to quickly lookup the groups from a given asset-id.
        asset_to_groups_id = [set() for _ in range(self._num_assets)]

        # Initialize empty maps from group-id's to sets of asset-id's.
        # This allows us to quickly lookup the assets from a given group-id.
        # NOTE: This is currently not being used for anything, but it
        # might be useful in the future, so it has just been disabled.
        # group_to_assets_id = [set() for _ in range(self._num_groups)]

        # Build the maps between asset-id's and group-id's.
        # For each asset-name and its associated list of group-names.
        for asset_name, list_group_names in asset_to_groups.items():
            # Lookup the asset-id from its name.
            asset_id = self._asset_name_to_id[asset_name]

            # For each group-name associated with this asset.
            for group_name in list_group_names:
                # Lookup the group-id from its name.
                group_id = self._group_name_to_id[group_name]

                # Add mapping from the asset-id to the group-id.
                asset_to_groups_id[asset_id].add(group_id)

                # Add mapping from the group-id to the asset-id.
                # NOTE: This is currently not being used for anything, but it
                # might be useful in the future, so it has just been disabled.
                # group_to_assets_id[group_id].add(asset_id)

        # Convert list-of-sets to list-of-numpy-arrays.
        self._asset_to_groups_np = \
            [np.fromiter(s, dtype=int) for s in asset_to_groups_id]

        # NOTE: This is currently not being used for anything, but it
        # might be useful in the future, so it has just been disabled.
        # self._group_to_assets_np = \
        #     [np.fromiter(s, dtype=int) for s in group_to_assets_id]

        # Convert list-of-numpy-arrays to numba-list-of-numpy-arrays.
        # This is the format expected by the function _constrain_weights
        # for optimal performance when using Numba Jit compilation.
        self._asset_to_groups_nb = nb.typed.List(self._asset_to_groups_np)

        # NOTE: This is currently not being used for anything, but it
        # might be useful in the future, so it has just been disabled.
        # self._group_to_assets_nb = nb.typed.List(self._group_to_assets_np)

    @staticmethod
    def _check_args(asset_names, group_names,
                    asset_to_groups, group_lim_pos, group_lim_neg):
        """Check the class-arguments are valid."""
        # Check type of asset_names.
        if not isinstance(asset_names, list):
            msg = 'Argument \'asset_names\' must be a list.'
            raise TypeError(msg)

        # Check type of group_names.
        if not isinstance(group_names, list):
            msg = 'Argument \'group_names\' must be a list.'
            raise TypeError(msg)

        # Sorted and unique group-names.
        group_names_unique = sorted(set(group_names))

        # Check type of asset_to_groups.
        if not isinstance(asset_to_groups, dict):
            msg = 'Argument \'asset_to_groups\' must be a dict.'
            raise TypeError(msg)

        # Check type of group_lim_pos.
        if not (group_lim_pos is None or isinstance(group_lim_pos, pd.Series)):
            msg = 'Argument \'group_lim_pos\' must be None or a Pandas Series.'
            raise TypeError(msg)

        # Check type of group_lim_neg.
        if not (group_lim_neg is None or isinstance(group_lim_neg, pd.Series)):
            msg = 'Argument \'group_lim_neg\' must be None or a Pandas Series.'
            raise TypeError(msg)

        # Check both group-limits are not None.
        if group_lim_pos is None and group_lim_neg is None:
            msg = 'Arguments \'group_lim_pos\' and \'group_lim_neg\' ' \
                  'cannot both be None.'
            raise ValueError(msg)

        # Check the positive group-limits are valid.
        if group_lim_pos is not None:
            # Check the index matches the group-names.
            if sorted(group_lim_pos.index) != group_names_unique:
                msg = 'Argument \'group_lim_pos\' has the wrong group-names.'
                raise ValueError(msg)

            # Check the group-limits are positive and non-zero.
            if np.any(group_lim_pos <= 0.0):
                msg = 'Argument \'group_lim_pos\' must be positive and non-zero.'
                raise ValueError(msg)

            # Check the group-limits are not NaN-values (Not-a-Number).
            if np.any(np.isnan(group_lim_pos)):
                msg = 'Argument \'group_lim_pos\' cannot have NaN values. ' \
                      'Use \'np.inf\' to disable some of the group-limits.'
                raise ValueError(msg)

        # Check the negative group-limits are valid.
        if group_lim_neg is not None:
            # Check the index matches the group-names.
            if sorted(group_lim_neg.index) != group_names_unique:
                msg = 'Argument \'group_lim_neg\' has the wrong group-names.'
                raise ValueError(msg)

            # Check the group-limits are positive and non-zero.
            if np.any(group_lim_neg >= 0.0):
                msg = 'Argument \'group_lim_neg\' must be negative and non-zero.'
                raise ValueError(msg)

            # Check the group-limits are not NaN-values (Not-a-Number).
            if np.any(np.isnan(group_lim_neg)):
                msg = 'Argument \'group_lim_neg\' cannot have NaN values. ' \
                      'Use \'-np.inf\' to disable some of the group-limits.'
                raise ValueError(msg)

    def constrain_weights(self, weights_org, fix_input=True,
                          log=False, max_iter=100, tol=1e-6):
        """
        Adjust the input weights so all the group-constraints are satisfied.
        This will only lower the portfolio weights and not increase them.

        The validity of the results are automatically checked. If `log==True`
        then the adjusted weights for all iterations are checked, so it is
        recommended to only use `log==True` if you really need to see the log.

        :param weights_org:
            Pandas Series with the portfolio weights to be adjusted.
            The asset-names in the index of `weights_org` must match the ones
            provided in the arg `asset_to_groups` when creating the solver.

        :param fix_input:
            Boolean whether to fix NaN and inf portfolio weights in the input.

        :param log:
            Boolean whether to return all iterations of the weights (True),
            or just the final weights (False). This is useful for debugging.

        :param max_iter:
            Integer with the max number of iterations for the algorithm.

        :param tol:
            Float with the convergence tolerance for aborting the algorithm,
            when all the weight-adjustments are below this number.
            This should be set to a small positive number such as 1e-6.

        :raises:
            - `TypeError` if `weights` is not a Pandas Series.
            - `ValueError` if `weights` has the wrong length.
            - `KeyError` if `weights` contains invalid asset-names.
            - `RuntimeError` if the adjusted weights are invalid.

        :return:
            - If `log==False` Pandas Series with adjusted portfolio weights.
            - If `log==True` Pandas DataFrame with log of adjusted weights.
        """
        # Save original weights assumed to be a Pandas Series for later use.
        weights_org_pd = weights_org

        # Check and convert input weights from Pandas Series to Numpy arrays.
        weights_org = self._convert_input_weights(weights=weights_org,
                                                  fix_input=fix_input)

        # Run algorithm for constraining the weights.
        # Note: This always returns a 2-dim Numpy array, even if log==False.
        weights_new = \
            _constrain_weights(weights_org=weights_org,
                               num_groups=self._num_groups, max_iter=max_iter,
                               asset_to_groups_nb=self._asset_to_groups_nb,
                               group_lim_pos=self._group_lim_pos, log=log,
                               group_lim_neg=self._group_lim_neg, tol=tol)

        # Were the results of all iterations logged?
        if not log:
            # Only use first row of 2-dim Numpy array.
            weights_new = weights_new[0]

            # Check the original and new portfolio weights are consistent, so
            # they have the same signs and abs(weights_new) <= abs(weights_org)
            check_weights(weights_org=weights_org, weights_new=weights_new)

            # Check the group constraints are satisfied.
            self._check_constraints(weights=weights_new)

            # Convert the adjusted weights to a Pandas Series.
            weights_new = pd.Series(data=weights_new, index=self._asset_names)
        else:
            # Check all iterations of the new portfolio weights, except the
            # first iteration because they are the original weights.
            for w_new in weights_new[1:]:
                # Check the original and new portfolio weights are consistent.
                check_weights(weights_org=weights_org, weights_new=w_new)

                # Check the group constraints are satisfied.
                self._check_constraints(weights=w_new)

            # Convert the adjusted weights to a Pandas DataFrame.
            weights_new = pd.DataFrame(data=weights_new,
                                       columns=self._asset_names)

        # Reorder the asset-names to match the original input weights.
        weights_new = weights_new[weights_org_pd.index]

        return weights_new

    def group_sums(self, weights, fix_input=True):
        """
        Calculate the group-sums for the given portfolio weights.

        :param weights:
            Either a Pandas Series with a single set of portfolio weights,
            or a Pandas DataFrame where each row is a set of portfolio weights.

        :param fix_input:
            Boolean whether to fix NaN and inf portfolio weights in the input.

        :raises:
            - `TypeError` if `weights` is not a Pandas Series.
            - `ValueError` if `weights` has the wrong length.
            - `KeyError` if `weights` contains invalid asset-names.

        :return:
            - Pandas Series / DataFrame with the POSITIVE group-sums.
            - Pandas Series / DataFrame with the NEGATIVE group-sums.
        """
        if isinstance(weights, pd.Series):
            # The input is just a single set of portfolio weights.

            # Convert input weights from Pandas Series to Numpy arrays.
            weights = self._convert_input_weights(weights=weights,
                                                  fix_input=fix_input)

            # Calculate the group-sums for both positive and negative weights.
            group_sums_pos, group_sums_neg = \
                _group_sums(weights=weights, num_groups=self._num_groups,
                            asset_to_groups_nb=self._asset_to_groups_nb)

            # Convert from Numpy arrays to Pandas Series.
            group_sums_pos = pd.Series(data=group_sums_pos, index=self._group_names)
            group_sums_neg = pd.Series(data=group_sums_neg, index=self._group_names)

        elif isinstance(weights, pd.DataFrame):
            # The input is a list of different sets of portfolio weights that
            # will be processed in turn by calling this function recursively,
            # and then finally combining the results.

            # Initialize lists for the results of the recursive function calls.
            group_sums_pos = []
            group_sums_neg = []

            # For each row of weights in the input.
            for _, weights_row in weights.iterrows():
                # Recursively call this function with the weights for this row.
                grp_sums_pos, grp_sums_neg = \
                    self.group_sums(weights=weights_row, fix_input=fix_input)

                # Save the results for later.
                group_sums_pos.append(grp_sums_pos)
                group_sums_neg.append(grp_sums_neg)

            # Convert the results of all the recursive calls to DataFrames.
            group_sums_pos = pd.DataFrame(group_sums_pos)
            group_sums_neg = pd.DataFrame(group_sums_neg)

        return group_sums_pos, group_sums_neg

    def group_ratios(self, weights, fix_input=True):
        """
        Calculate the group-ratios between the group-limits and group-sums
        for the given portfolio weights. The ratios will all be positive.

        A group-ratio below 1.0 means that the group-sum is too high so the
        portfolio weights for that group must be decreased by multiplying
        with that group-ratio. A group-ratio above 1.0 means that the
        group-sum is below the group-limit, so it is possible to increase
        the portfolio weights for that group by multiplying with that
        group-ratio, without exceeding the limit for that particular group,
        although it may then exceed the limit for other groups, which is
        handled properly in the function `constrain_weights`.

        A group-ratio will be `np.inf` (infinity) if the group-limit was inf
        or the group-sum was zero.

        :param weights:
            Either a Pandas Series with a single set of portfolio weights,
            or a Pandas DataFrame where each row is a set of portfolio weights.

        :param fix_input:
            Boolean whether to fix NaN and inf portfolio weights in the input.

        :return:
            - Pandas Series / DataFrame with the POSITIVE group-ratios.
            - Pandas Series / DataFrame with the NEGATIVE group-ratios.
        """
        # Calculate the group-sums for both positive and negative weights.
        group_sums_pos, group_sums_neg = \
            self.group_sums(weights=weights, fix_input=fix_input)

        # Calculate the group-ratios for POSITIVE weights.
        # If the group-sum is zero, then the ratio is inf.
        # Because the group-sums are either Pandas Series or DataFrame,
        # it supports division-by-zero without raising an exception.
        if self._group_lim_pos is not None:
            group_ratios_pos = self._group_lim_pos / group_sums_pos
        else:
            group_ratios_pos = None

        # Calculate the group-ratios for NEGATIVE weights.
        # The ratios should all be positive, and there should
        # not be any need to take the absolute values.
        if self._group_lim_neg is not None:
            group_ratios_neg = self._group_lim_neg / group_sums_neg
        else:
            group_ratios_neg = None

        return group_ratios_pos, group_ratios_neg

    def plot_log(self, weights_org, weights_new_log, all_xticks=False,
                 rasterized=True, figsize=(10, 12)):
        """
        Plot the weight-ratios and group-ratios for a log of adjusted weights.
        This makes it easy to see how the algorithm improves the ratios for
        each iteration.

        :param weights_org:
            Pandas Series with the original portfolio weights.

        :param weights_new_log:
            Pandas DataFrame with a log of adjusted portfolio weights.

        :param all_xticks:
            Boolean whether to have a tick on the x-axis for all iterations.

        :param rasterized:
            Boolean whether to turn the lines in the plot into pixels.
            If you want to save the plot to a file, then this makes the file
            much smaller when there are a lot of assets and groups.

        :param figsize:
            Tuple with the figure-size which is passed to Matplotlib.

        :return:
            Matplotlib Figure object.
        """
        # Check type of original weights.
        if not isinstance(weights_org, pd.Series):
            msg = 'Argument \'weights_org\' must be a Pandas Series.'
            raise TypeError(msg)

        # Check type of new weights log.
        if not isinstance(weights_new_log, pd.DataFrame):
            msg = 'Argument \'weights_new_log\' must be a Pandas DataFrame.'
            raise TypeError(msg)

        # Calculate the inverted weight-ratios.
        weight_ratios = weights_new_log / weights_org

        # Calculate group-ratios.
        group_ratios_pos, group_ratios_neg = \
            self.group_ratios(weights=weights_new_log)

        # Count the number of rows we need in the plot.
        num_rows = 1
        if group_ratios_pos is not None:
            num_rows += 1
        if group_ratios_neg is not None:
            num_rows += 1

        # Create a new plot with several rows of sub-plots.
        fig, axs = plt.subplots(num_rows, 1, sharex=True, sharey=False,
                                figsize=figsize)

        # Common arguments for the plotting functions.
        args = dict(grid=True, legend=False, rasterized=rasterized)

        # Set common x-ticks for all sub-plots.
        if all_xticks:
            # Set an x-tick for all iterations of the algorithm.
            axs[0].set_xticks(range(len(weight_ratios)))
        else:
            # Use the default x-ticks but ensure they are integers.
            axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        # Set x-axis label only for bottom sub-plot.
        xlabel = 'Iteration Number (Zero are the original weights)'
        axs[-1].set_xlabel(xlabel)

        # Plot weight-ratios.
        title = 'Weight Ratios (Inverted) = New Weight / Original Weight'
        weight_ratios.plot(ax=axs[0], color='blue', title=title, alpha=0.1,
                           **args)
        axs[0].set_ylabel('Weight Ratio (Inv.)')

        # Plot POSITIVE group-ratios.
        if group_ratios_pos is not None:
            title = 'Positive Group Ratios = Pos. Group Limit / Pos. Group Sum'
            group_ratios_pos.plot(ax=axs[1], color='green', title=title, **args)
            axs[1].set_ylabel('Pos. Group Ratio')

        # Plot NEGATIVE group-ratios.
        if group_ratios_neg is not None:
            title = 'Negative Group Ratios = Neg. Group Limit / Neg. Group Sum'
            group_ratios_neg.plot(ax=axs[2], color='red', title=title, **args)
            axs[2].set_ylabel('Neg. Group Ratio')

        # For all sub-plots.
        for ax in axs:
            # Plot a dotted horizontal line at value 1.0.
            ax.axhline(1.0, color='black', linestyle='dotted')

        return fig

    def _convert_input_weights(self, weights, fix_input=True):
        """
        Check the portfolio weights are valid,
        and convert from Pandas Series to Numpy array.

        :param weights:
            Pandas Series with portfolio weights.

        :param fix_input:
            Boolean whether to fix NaN and inf portfolio weights in the input.

        :raises:
            - `TypeError` if `weights` is not a Pandas Series.
            - `ValueError` if `weights` has the wrong length.
            - `KeyError` if `weights` contains invalid asset-names.

        :return:
            Numpy array with a copy of the weights, where the assets are
            ordered in the same way as the internal data-structures.
        """
        # Check weights is a Pandas Series.
        if not isinstance(weights, pd.Series):
            msg = 'Argument \'weights\' is not a Pandas Series.'
            raise TypeError(msg)

        # Check the number of weights is correct.
        if len(weights) != len(self._asset_names):
            msg = 'len(\'weights\')={} but expected {}.'
            msg = msg.format(len(self._asset_names))
            raise ValueError(msg)

        # Convert weights to Numpy array, and ensure ordering
        # is the same as used in the internal data-structures.
        # Note: This is a copy of the data because of the reordering.
        weights_np = weights[self._asset_names].to_numpy()

        # Ensure the weights are valid.
        if fix_input:
            # The data was already copied above so just update it inplace.
            np.nan_to_num(weights_np, nan=0.0, copy=False)

        return weights_np

    def _check_constraints(self, weights, tol=1e-6):
        """
        Check that the group-constraints are satisfied.

        :param weights:
            Numpy array with the portfolio weights.

        :param tol:
            Error tolerance when comparing group-ratios to 1.0 to account for
            small floating-point rounding errors. This should be a small
            positive number such as 1e-6. Note that it is not directly related
            to the `tol` argument used in the function `constrain_weights`.

        :raises:
            `RuntimeError` if group constraints are not satisfied.

        :return:
            None
        """
        # Calculate the group-sums for both positive and negative weights.
        group_sums_pos, group_sums_neg = \
            _group_sums(weights=weights, num_groups=self._num_groups,
                        asset_to_groups_nb=self._asset_to_groups_nb)

        # Calculate the group-ratios for both positive and negative weights.
        group_ratios_pos, group_ratios_neg = \
            _group_ratios(num_groups=self._num_groups,
                          group_sums_pos=group_sums_pos,
                          group_sums_neg=group_sums_neg,
                          group_lim_pos=self._group_lim_pos,
                          group_lim_neg=self._group_lim_neg)

        # Min group-ratios for positive and negative weights.
        # These should all be positive so there is no need to use np.abs()
        # Ignore NaN even though it should never occur.
        min_group_ratio_pos = np.nanmin(group_ratios_pos)
        min_group_ratio_neg = np.nanmin(group_ratios_neg)

        # Overall min group-ratio.
        # Note: We need to convert to Numpy array with dtype=float to convert
        # a potential None to np.nan, otherwise the min() operation will fail.
        arr = np.array([min_group_ratio_pos, min_group_ratio_neg], dtype=float)
        min_group_ratio = np.nanmin(arr)

        # Check group constraints are satisfied. The min group-ratio must be
        # above 1.0 (with small tolerance for float rounding error), which
        # means that all the group-sums are smaller than their group-limits.
        if min_group_ratio < 1.0 - tol:
            msg = 'Group constraints not satisfied: ' + \
                  f'min_group_ratio={min_group_ratio}, tol={tol:.0e}'
            raise RuntimeError(msg)

###############################################################################
