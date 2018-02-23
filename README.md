# RaSCaLS
Range, Shift, Cluster and Link Scheme (RaSCaLS)

[![Build Status](https://www.travis-ci.com/moeyensj/RaSCaLS.svg?token=sWjpnqPgpHyuq3j7qPuj&branch=master)](https://www.travis-ci.com/moeyensj/RaSCaLS)
[![Coverage Status](https://coveralls.io/repos/github/moeyensj/RaSCaLS/badge.svg?t=Eu0phN)](https://coveralls.io/github/moeyensj/RaSCaLS)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Installation
To install pre-requisite software using anaconda: 

```conda install -c defaults -c conda-forge --file requirements.txt```

To install pre-requisite software using pip:

```pip install -r requirements.txt```

Once pre-requisites have been installed:

```python setup.py install```

## Testing Installation

Using pytest (with coveralls):

```pytest rascals --cov=rascals```

Or using setuptools:

```python setup.py test```
