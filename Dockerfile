FROM continuumio/miniconda3

# Set shell to bash
SHELL ["/bin/bash", "-c"]

# Update apps
RUN apt-get update \
	&& apt-get upgrade -y

# Update conda
RUN conda update -n base -c defaults conda

# Download THOR and install
RUN mkdir projects \
	&& cd projects \
	&& git clone https://github.com/moeyensj/thor.git --depth=1 \
	&& cd thor \
	&& conda install -c defaults -c conda-forge -c astropy -c moeyensj --file requirements.txt python=3.8 --y \
	&& python -m ipykernel install --user --name thor_py38 --display-name "THOR (Python 3.8)" \
	&& python setup.py install
