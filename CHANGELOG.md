# Changes to the InvestOps Python package

## Version 0.4.0 (2022-03-19)

New features:

- Class `StockForecast` (`stock_forecast.py`)
- Functions `rel_change` and `mean_rel_change` (`rel_change.py`)


## Version 0.3.0 (2022-02-12)

New features:

- Portfolio group constraints (`group_constraints.py`)
- Generate random portfolio groups (`random.py`)
- Function `random.rand_where`

Changes:

- Selection of parallel vs. serial execution in `diversify.py`

Moved:

- `rand_weights_uniform` to `rand_uniform`
- `rand_weights_normal` to `rand_normal`
- `diversify._check_weights` to `check.check_weights`


## Version 0.2.0 (2022-01-17)

New features:

- Portfolio diversification for sparse corr. matrix (`diversify_sparse.py`)
- Functions for sparse matrices (`sparse.py`)
- Functions for removing zero/small portfolio weights (`remove_weights.py`)

Fixes:

- Parallel "race condition" in function `diversify._update_weights`


## Version 0.1.0 (2021-10-23)

First beta version with the following features:

- Portfolio diversification (`diversify.py`)
- Portfolio weight normalization (`normalize.py`)
- Random generation of portfolio weights and correlation matrix (`random.py`)
- Check and fix correlation matrix (`check.py`)
- Linear mapping (`maps.py`)
