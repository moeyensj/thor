FROM continuumio/miniconda3
MAINTAINER Joachim Moeyens <moeyensj@gmail.com>

# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Update apps
RUN apt-get update \
	&& apt-get upgrade -y

# Update conda
RUN conda update -n base -c defaults conda

# Download THOR
RUN mkdir projects \
	&& cd projects \
	&& git clone https://github.com/moeyensj/thor.git --depth=1

# Create Python 3.6 conda environment and install requirements, then install THOR
RUN cd projects/thor \
	&& conda install -c defaults -c conda-forge -c astropy --file requirements.txt python=3.6 --y \
	&& python -m ipykernel install --user --name thor_py36 --display-name "THOR (Python 3.6)" \
	&& python setup.py install
