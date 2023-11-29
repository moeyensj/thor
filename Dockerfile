FROM ubuntu:latest

# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Update system dependencies
RUN apt-get update \
	&& apt-get upgrade -y \
	&& apt-get install -y curl gfortran git liblapack-dev make pip python3.11 python-is-python3 unzip

# Upgrade pip to the latest version and install pre-commit
RUN pip install --upgrade pip pre-commit

# Download openorb data files and set the environment variable
RUN curl -fL -o /tmp/oorb_data.zip \
    "https://github.com/B612-Asteroid-Institute/oorb/releases/download/v1.2.1a1.dev2/oorb_data.zip"
RUN unzip -d /opt/oorb_data /tmp/oorb_data.zip
ENV OORB_DATA=/opt/oorb_data

# Update OBSCODE.dat
RUN cd $OORB_DATA \
	&& curl https://www.minorplanetcenter.net/iau/lists/ObsCodes.html -o ObsCodes.html \
	&& sed -e '2d' ObsCodes.html | grep -v "<" > OBSCODE.dat \
	&& rm -f ObsCodes.html

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
