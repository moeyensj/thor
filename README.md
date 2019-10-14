![banner](docs/banner.png)
# THOR
Tracklet-less Heliocentric Orbit Recovery  
[![Build Status](https://dev.azure.com/moeyensj/thor/_apis/build/status/moeyensj.thor?branchName=master)](https://dev.azure.com/moeyensj/thor/_build/latest?definitionId=2&branchName=master)
[![Build Status](https://www.travis-ci.com/moeyensj/thor.svg?token=sWjpnqPgpHyuq3j7qPuj&branch=master)](https://www.travis-ci.com/moeyensj/thor)
[![Coverage Status](https://coveralls.io/repos/github/moeyensj/thor/badge.svg?branch=master&t=pdSkQA)](https://coveralls.io/github/moeyensj/thor?branch=master)
[![Docker Pulls](https://img.shields.io/docker/pulls/moeyensj/thor)](https://hub.docker.com/r/moeyensj/thor)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Installation

Warning: The THOR code is still in very active development. 

The corresponding notebook repository can be found at: https://github.com/moeyensj/thor_notebooks

### Source
Clone this repository using either `ssh` or `https`.

To create a `conda` environment in which to run the code:  
```conda create -n thor_py36 -c defaults -c conda-forge -c astropy --file requirements.txt python=3.6```

To install pre-requisite software into any `conda` enviroment:  
```conda activate env```
```conda install -c defaults -c conda-forge --file requirements.txt```

To install pre-requisite software using pip:  
```pip install -r requirements.txt```

Once pre-requisites have been installed:  
```python setup.py install```

### Docker

A Docker container with the latest version of the code can be pulled using:  
```docker pull moeyensj/thor:latest```

To run the container:  
```docker run -it moeyensj/thor:latest```
