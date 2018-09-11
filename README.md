![banner](docs/images/banner.png)
# THOR
Tracklet-less Heliocentric Orbit Recovery

[![Build Status](https://www.travis-ci.com/moeyensj/thor.svg?token=sWjpnqPgpHyuq3j7qPuj&branch=master)](https://www.travis-ci.com/moeyensj/thor)
[![Coverage Status](https://coveralls.io/repos/github/moeyensj/thor/badge.svg?t=Eu0phN)](https://coveralls.io/github/moeyensj/thor)
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

```pytest thor --cov=thor```

Or using setuptools:

```python setup.py test```
