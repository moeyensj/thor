FROM continuumio/miniconda3

# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Update system dependencies
RUN apt-get update \
	&& apt-get upgrade -y

# Update conda
RUN conda update -n base -c defaults conda
RUN conda install pip python=3.10

# Upgrade pip to the latest version and install pre-commit
RUN pip install --upgrade pip pre-commit

# Install openorb from conda
RUN conda install -c defaults -c conda-forge openorb --y
ENV OORB_DATA /opt/conda/share/openorb

# Install pre-commit hooks (before THOR is installed to cache this step)
RUN mkdir /code/
COPY .pre-commit-config.yaml /code/
WORKDIR /code/
RUN git init . \
	&& git add .pre-commit-config.yaml \
	&& pre-commit install-hooks \
	&& rm -rf .git

# Install THOR
ADD . /code/
RUN SETUPTOOLS_SCM_PRETEND_VERSION=1 pip install -e .[tests]
