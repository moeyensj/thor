FROM ubuntu:latest

# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Update system dependencies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y curl gfortran git liblapack-dev make pip python3.11 python-is-python3 unzip wget

# Upgrade pip to the latest version and install pre-commit
RUN pip install --upgrade pip pre-commit
RUN pip install --upgrade cython==0.29.36 setuptools setuptools_scm
RUN chmod 777 /opt

# Install numpy
RUN git clone https://github.com/numpy/numpy.git /opt/numpy
RUN cd /opt/numpy && git checkout v1.24.4 && git submodule update --init
RUN cd /opt/numpy && python3 setup.py build --cpu-baseline=native install


# Install openorb
# TODO: We need a more robust way to be appropriately natively compiled pyoorb installed
# including data file generation
RUN git clone https://github.com/B612-Asteroid-Institute/oorb.git /opt/oorb
RUN cd /opt/oorb && git checkout fork
RUN cd /opt/oorb && ./configure gfortran opt --with-pyoorb --with-f2py=/usr/local/bin/f2py --with-python=python3
# Add '-march=native' to compiler options by running a sed
# script directly on the Makefile.includse file. This is a
# hack to get around the fact that the configure script
# doesn't support this option.
RUN sed -i 's/FCOPTIONS = .*/FCOPTIONS = $(FCOPTIONS_OPT_GFORTRAN) -march=native/g' /opt/oorb/Makefile.include
# --no-build-isolation is needed because we need to ensure we use
# the same version of numpy as the one we compiled previously so
# that it matches the version of f2py we passed in to ./configure.
RUN pip install --no-build-isolation -v /opt/oorb

# Generate the data files
RUN cd /opt/oorb && make ephem
RUN cd /opt/oorb/data && ./getBC430
RUN cd /opt/oorb/data && ./updateOBSCODE
ENV OORBROOT=/opt/oorb
ENV OORB_DATA=/opt/oorb/data

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
RUN SETUPTOOLS_SCM_PRETEND_VERSION=1 pip install -e .[tests,dev]
