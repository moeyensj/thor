# THOR - Tracklet-less Heliocentric Orbit Recovery

<img src="docs/banner.png" width="400" height="200">


[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://img.shields.io/badge/Python-3.9%2B-blue)[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)[![DOI](https://zenodo.org/badge/116747066.svg)](https://zenodo.org/badge/latestdoi/116747066)[![docker - Build, Lint, and Test](https://github.com/moeyensj/thor/actions/workflows/docker-build-lint-test.yml/badge.svg)](https://github.com/moeyensj/thor/actions/workflows/docker-build-lint-test.yml)[![conda - Build, Lint, and Test](https://github.com/moeyensj/thor/actions/workflows/conda-build-lint-test.yml/badge.svg)](https://github.com/moeyensj/thor/actions/workflows/conda-build-lint-test.yml)[![pip - Build, Lint, Test, and Coverage](https://github.com/moeyensj/thor/actions/workflows/pip-build-lint-test-coverage.yml/badge.svg)](https://github.com/moeyensj/thor/actions/workflows/pip-build-lint-test-coverage.yml)[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)[![Coverage Status](https://coveralls.io/repos/github/moeyensj/thor/badge.svg?branch=main)](https://coveralls.io/github/moeyensj/thor?branch=main)[![Docker Pulls](https://img.shields.io/docker/pulls/moeyensj/thor)](https://hub.docker.com/r/moeyensj/thor)[![Anaconda-Server Badge](https://anaconda.org/moeyensj/thor/badges/version.svg)](https://anaconda.org/moeyensj/thor)[![Anaconda-Server Badge](https://anaconda.org/moeyensj/thor/badges/platforms.svg)](https://anaconda.org/moeyensj/thor)[![Anaconda-Server Badge](https://anaconda.org/moeyensj/thor/badges/downloads.svg)](https://anaconda.org/moeyensj/thor)

## Table of Contents

- [Introduction](#introduction)
- [Development Status](#development-status)
- [Installation](#installation)
    - [Anaconda](#anaconda)
    - [Docker](#docker)
    - [Source](#source)
- [Contributing to THOR](#contributing-to-thor)
    - [Ways to Contribute](#ways-to-contribute)
    - [Getting Started](#getting-started)
    - [Contributing Code](#contributing-code)
    - [Reporting Bugs](#reporting-bugs)
    - [Documentation](#documentation)
    - [Testing](#testing)
    - [Review Process](#review-process)
- [Reporting Issues](#reporting-issues)
- [Acknowledgements](#acknowledgments)
- [License](#license)
- [Citation](#citation)




## Introduction

THOR (Tracklet-less Heliocentric Orbit Recovery) is an active development project aimed at revolutionizing heliocentric orbit recovery. This README provides essential information on installing/using and contributing to THOR.

## Development Status

**Warning: THOR is still in very active development.**

The latest "stable" version is [v1.2](https://github.com/moeyensj/thor/releases/tag/v1.2). The code on the main branch is currently being used to develop THOR v2.0 and is not guaranteed to be stable. We anticipate that v2.0 will be the one most useful to the community and we aim for it to be released by the end of 2023. THOR v2.0 is a complete re-write of the THOR code primarily designed to enable it for use as a service on the [Asteroid, Discovery, Analysis and Mapping (ADAM) platform](https://b612.ai/). The primary goal of v2.0 is to enable THOR to work at scale on many small cloud-hosted VMs. The secondary goal of v2.0 is to add changes that will work towards enabling the linking of NEOs (THOR is currently configured to work on the Main Belt and outwards).


## Installation

The corresponding notebook repository can be found at: [THOR Notebooks](https://github.com/moeyensj/thor_notebooks)

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

**If you would like to run Jupyter Notebook or Juptyter Lab with THOR please see the installation instructions in the THOR notebooks repository.**


# Contributing to THOR

We welcome and appreciate contributions from the community to help make THOR even better. Whether you're interested in improving code, fixing bugs, enhancing documentation, or providing feedback, your contributions are valuable to us. Please take a moment to review the following guidelines before getting started.

---
## Ways to Contribute

There are various ways to contribute to THOR:

1. **Code Contributions:** Help us improve THOR by adding features, fixing bugs, or optimizing code.
2. **Documentation:** Enhance our documentation for clarity and user-friendliness.
3. **Bug Reports:** Report any issues or bugs you encounter while using THOR.
4. **Testing:** Write tests and ensure existing tests pass to maintain code quality.
5. **Feedback:** Share your experiences, suggestions, and feedback.
6. **Community Support:** Assist others by answering questions and engaging in discussions.

## Getting Started

To start contributing to THOR, follow these steps:

1. Fork the THOR repository to your GitHub account.
2. Clone your forked repository to your local development environment.
3. Create a new branch for your contributions (e.g., `feature/new-feature` or `bugfix/fix-issue`).
4. Implement your changes or additions.
5. Commit your changes with clear and concise messages.
6. Push your changes to your forked repository.
7. Submit a pull request to the main THOR repository.

## Contributing Code

When contributing code, please adhere to these guidelines:

- Write clear and concise code with meaningful variable and function names.
- Include comments when necessary to explain complex logic.
- Ensure your code follows our coding standards and style.
- Thoroughly test your changes and add new tests if applicable.
- Maintain backward compatibility when making changes.
- Update the documentation to reflect code changes.
- Keep pull requests focused on a single task or issue.

## Reporting Bugs

If you encounter a bug or issue while using THOR, please report it by opening a new issue on the GitHub repository. Include:

- A descriptive title.
- A detailed description of the issue with steps to reproduce it.
- Your operating system and version.
- The version of THOR you are using (if applicable).

## Documentation

Improving documentation is a valuable contribution. If you find areas that need enhancement or wish to add new documentation, please feel free to do so. Documentation updates may include changes to the README, creation of new guides, and clarification of existing content.

## Testing

We encourage contributors to write tests for new code and ensure existing tests pass. Running tests before submitting a pull request helps maintain code quality and reliability.

## Review Process

All contributions, whether code changes, documentation improvements, or bug reports, will undergo review by maintainers and contributors. Feedback and suggestions may be provided during the review process.

---

## Reporting Issues

If you encounter any issues or have suggestions for improvements, please report them on our [GitHub Issues](https://github.com/moeyensj/thor/issues) page. We appreciate your feedback and will work diligently to address any problems.

## Acknowledgments

We would like to thank all contributors and supporters of the THOR project. Your contributions and feedback are invaluable in driving the development of THOR.

## License

THOR is licensed under the [BSD 3-Clause License](LICENSE). By using or contributing to this project, you agree to abide by the terms specified in the license.

## Citation

If you find THOR useful for your research or work, please consider citing it. You can use the following DOI:
[![DOI](https://zenodo.org/badge/116747066.svg)](https://zenodo.org/badge/latestdoi/116747066) 

---

Thank you for choosing THOR for your heliocentric orbit recovery needs. We're excited to have you on board and look forward to the amazing things we can accomplish together. Happy coding!

