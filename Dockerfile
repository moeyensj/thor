FROM continuumio/miniconda3

# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Update system dependencies
RUN apt-get update \
	&& apt-get upgrade -y

# Update conda
RUN conda update -n base -c defaults conda

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install openorb from conda
RUN conda install -c defaults -c conda-forge openorb --y

# Install THOR
RUN mkdir /code/
ADD . /code/
WORKDIR /code/
RUN pip install -e .[tests]

# Install pre-commit hooks
RUN pre-commit install
