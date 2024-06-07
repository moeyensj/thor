![banner](docs/banner.png)
# THOR
Tracklet-less Heliocentric Orbit Recovery  

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://img.shields.io/badge/Python-3.9%2B-blue)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/116747066.svg)](https://zenodo.org/badge/latestdoi/116747066)  
[![docker - Build, Lint, and Test](https://github.com/moeyensj/thor/actions/workflows/docker-build-lint-test.yml/badge.svg)](https://github.com/moeyensj/thor/actions/workflows/docker-build-lint-test.yml)
[![conda - Build, Lint, and Test](https://github.com/moeyensj/thor/actions/workflows/conda-build-lint-test.yml/badge.svg)](https://github.com/moeyensj/thor/actions/workflows/conda-build-lint-test.yml)
[![pip - Build, Lint, Test, and Coverage](https://github.com/moeyensj/thor/actions/workflows/pip-build-lint-test-coverage.yml/badge.svg)](https://github.com/moeyensj/thor/actions/workflows/pip-build-lint-test-coverage.yml)  
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Coverage Status](https://coveralls.io/repos/github/moeyensj/thor/badge.svg?branch=main)](https://coveralls.io/github/moeyensj/thor?branch=main)
[![Docker Pulls](https://img.shields.io/docker/pulls/moeyensj/thor)](https://hub.docker.com/r/moeyensj/thor)  
[![Anaconda-Server Badge](https://anaconda.org/moeyensj/thor/badges/version.svg)](https://anaconda.org/moeyensj/thor)
[![Anaconda-Server Badge](https://anaconda.org/moeyensj/thor/badges/platforms.svg)](https://anaconda.org/moeyensj/thor)
[![Anaconda-Server Badge](https://anaconda.org/moeyensj/thor/badges/downloads.svg)](https://anaconda.org/moeyensj/thor)


**Warning: THOR is still in very active development.**

The latest "stable" version is [v1.2](https://github.com/moeyensj/thor/releases/tag/v1.2). The code on the main branch is currently being used to develop THOR v2.0 and is not guaranteed to be stable. We anticipate that v2.0 will be the one most useful to the community and we aim for it to be released by the end of 2023. THOR v2.0 is a complete re-write of the THOR code primarily designed to enable it for use as a service on the [Asteroid, Discovery, Analysis and Mapping (ADAM) platform](https://b612.ai/). The primary goal of v2.0 is to enable THOR to work at scale on many small cloud-hosted VMs. The secondary goal of v2.0 is to add changes that will work towards enabling the linking of NEOs (THOR is currently configured to work on the Main Belt and outwards).  



## Table of Contents

1. What THOR Does
2. Installation
    a. Anaconda
    b. Docker
    c. Source
3. Running the Project
4. Motivation Behind Technologies

## What THOR Does
THOR stands for Tracklet-less Heliocentric Orbit Recovery. Let's break it down.
 * Tracklet: A short track between frames of consecutive observations in time
 * Heliocentric: A model of the solar system having the sun as the center
 * Orbit: the curved trajectory of an object around a celestial object
 * Recovery: Regaining knowledge of information that was lost

Esentially, this project helps us to learn more about different objects orbiting in a sun-centered solar system, including but not limited to information regarding the object's time, velocity, range, and shifts in space. The project uses trained models and a wide range of datasets to accomplish this tracklet-less recovery, and with more test data to train the models, the better the program can predict the orbit data. 

## Installation

The corresponding notebook repository can be found at: https://github.com/moeyensj/thor_notebooks

The following installation paths are available:  
[Anaconda](#Anaconda)  
[Docker](#Docker)  
[Source](#Source)  

### Anaconda
`thor` can be downloaded directly from anaconda:  
```conda install -c moeyensj thor```

Or, if preferred, installed into its own environment via:  
```conda create -n thor_py38 -c moeyensj thor python=3.8```

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

### Source
Clone this repository using either `ssh` or `https`. Once cloned and downloaded, `cd` into the repository.

To install THOR in its own `conda` enviroment please do the following:  

```conda create -n thor_py38 -c defaults -c conda-forge -c astropy -c moeyensj --file requirements.txt python=3.8```  

Or, to install THOR in a pre-existing `conda` environment called `env`:  

```conda activate env```  
```conda install -c defaults -c conda-forge -c astropy -c moeyensj --file requirements.txt```  

Once pre-requisites have been installed using either one of the options above, then:  

```python setup.py install```

Or, if you are actively planning to develop or contribute to THOR, then:

```python setup.py develop --no-deps```

You should now be able to start Python and import THOR.
```
┌─(thor_py38)[moeyensj][±][main ✓][~/projects/thor]
└─▪ python
Python 3.8.8 (default, Apr 13 2021, 19:58:26)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import thor
>>>
```

## Running the Project
To run the project, follow the installation steps above. Then, navigate to your favorite python IDE and run runTHOR.py with relevant arguments as listed in the runTHOR.py file. 

## Motivation Behind Technologies

* Python: Rapid application development, high-level data structures built in to the language, simple
* Anaconda: Helps to create environment for various versions of Python (makes sharing/distributing the code easier)
* Docker: Allows for fast and efficient project delivery and updates

**If you would like to run Jupyter Notebook or Juptyter Lab with THOR please see the installation instructions in the THOR notebooks repository.**
