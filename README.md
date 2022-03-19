# InvestOps

[Original repository on GitHub](https://github.com/Hvass-Labs/InvestOps)

Original author is [Magnus Erik Hvass Pedersen](http://www.hvass-labs.org)


## Introduction

This is a Python package with simple and effective tools for investing,
including the following:

- Long-Term Stock Forecasting.
  ([Tutorial](https://github.com/Hvass-Labs/InvestOps-Tutorials/blob/master/Long-Term_Stock_Forecasting.ipynb))
  ([Paper](https://ssrn.com/abstract=3750775))

- Portfolio diversification using the so-called "Hvass Diversification"
  algorithm, which is extremely fast to compute and very robust to estimation
  errors in the correlation matrix.
  ([Tutorial](https://github.com/Hvass-Labs/InvestOps-Tutorials/blob/master/Portfolio_Diversification.ipynb))
  ([Paper](https://ssrn.com/abstract=3942552))

- Portfolio Group Constraints.
  ([Tutorial](https://github.com/Hvass-Labs/InvestOps-Tutorials/blob/master/Portfolio_Group_Constraints.ipynb))
  ([Paper](https://ssrn.com/abstract=4033243))

The InvestOps Python package is a distilled version of some of the research
from the [FinanceOps](https://github.com/Hvass-Labs/FinanceOps) project,
whose papers can be found on [SSRN](http://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=1993051)
and [GitHub](https://github.com/Hvass-Labs/Finance-Papers).


## Tutorials

[Tutorials](https://github.com/Hvass-Labs/InvestOps-Tutorials) are made as
Python Notebooks which can be modified and run entirely on the internet
or on your own computer.


## Installation

It is best to use a virtual environment when installing Python packages,
so you can easily delete the environment again if something goes wrong.
You write the following in a Linux terminal:

    virtualenv investops-env

Or you can use [Anaconda](https://www.anaconda.com/download) instead of a virtualenv:

    conda create --name investops-env python=3

Then you switch to the virtual environment:

    source activate investops-env

And then you can install the InvestOps package inside that virtual environment:
 
    pip install investops   

You can now import the InvestOps package in your Python program as follows:

    import investops as iv

    # Print the InvestOps version.
    print(iv.__version__)


## License (MIT)

This is published under the [MIT License](https://github.com/Hvass-Labs/InvestOps/blob/main/LICENSE)
which allows very broad use for both academic and commercial purposes.

You are very welcome to modify and use this source-code in your own project.
Please keep a link to the [original repository](https://github.com/Hvass-Labs/InvestOps).
