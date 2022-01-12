# Changes to the InvestOps Python package

## Version 0.2.0 (2021-???)

Added the following features:

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
