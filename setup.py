# -*- coding: utf-8 -*-
from setuptools import setup

# Also defined in investops/__init__.py and must be updated in both places.
MY_VERSION = '0.4.0'

setup(
    name='investops',
    packages=['investops'],
    version=MY_VERSION,
    description='Tools for investing in Python',
    long_description='InvestOps is a collection of Python tools for investing, '
                     'including the fast and robust \"Hvass Diversification\" '
                     'algorithm for diversifying an investment portfolio.',
    long_description_content_type="text/markdown",
    author='Magnus Erik Hvass Pedersen',
    author_email='my-first-name@hvass-labs.org',
    url='https://github.com/Hvass-Labs/InvestOps',
    license='MIT',
    keywords=['investing', 'portfolio optimization', 'hvass diversification'],
    install_requires=[
        'numpy',
        'pandas',
        'numba',
        'matplotlib',
    ],
)

