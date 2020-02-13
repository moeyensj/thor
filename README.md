![banner](docs/banner.png)
# THOR
Tracklet-less Heliocentric Orbit Recovery  
[![Build Status](https://dev.azure.com/moeyensj/thor/_apis/build/status/moeyensj.thor?branchName=master)](https://dev.azure.com/moeyensj/thor/_build/latest?definitionId=2&branchName=master)
[![Build Status](https://www.travis-ci.com/moeyensj/thor.svg?token=sWjpnqPgpHyuq3j7qPuj&branch=master)](https://www.travis-ci.com/moeyensj/thor)
[![Coverage Status](https://coveralls.io/repos/github/moeyensj/thor/badge.svg?branch=master&t=pdSkQA)](https://coveralls.io/github/moeyensj/thor?branch=master)
[![Docker Pulls](https://img.shields.io/docker/pulls/moeyensj/thor)](https://hub.docker.com/r/moeyensj/thor)  
[![Python 3.6](https://img.shields.io/badge/Python-3.6%2B-blue)](https://img.shields.io/badge/Python-3.6%2B-blue)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)  

## Installation

Warning: The THOR code is still in very active development. 

The corresponding notebook repository can be found at: https://github.com/moeyensj/thor_notebooks

We recommend installing the code along one of two installation paths: either a source code installation, or an installation via docker. 

### Source
Clone this repository using either `ssh` or `https`. Once cloned and downloaded, `cd` into the repository. 

To install THOR in its own `conda` enviroment please do the following:  

```conda create -n thor_py36 -c defaults -c conda-forge -c astropy --file requirements.txt python=3.6```  

Or, to install THOR in a pre-existing `conda` environment called `env`:  

```conda activate env```  
```conda install -c defaults -c conda-forge -c astropy --file requirements.txt```  

Or, to install pre-requisite software using `pip`:  

```pip install -r requirements.txt```

Once pre-requisites have been installed using either one of the tree options above, then:  

```python setup.py install```

You should now be able to start Python and import THOR. 
```
┌─(thor_py36)[~]
└─▪ python
Python 3.6.9 (default, Jul 30 2019, 19:07:31) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import thor
>>> 
```

### Docker

A Docker container with the latest version of the code can be pulled using:  

```docker pull moeyensj/thor:latest```

To run the container:  

```docker run -it moeyensj/thor:latest```

The THOR code is installed the /projects directory, and is by default also installed in the container's Python installation. 
To access the code in Python: 
```
(base) root@202110177eee:/# python
Python 3.6.9 |Anaconda, Inc.| (default, Jul 30 2019, 19:07:31) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import thor
>>> 
```

**If you would like to run Jupyter Notebook or Juptyter Lab with THOR please see the installation instructions in the THOR notebooks repository.**
