FROM python:3.11

# Set shell to bash
SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]

# Update system dependencies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y pip git wget tar gfortran liblapack-dev

# Install openorb dependencies
RUN wget -P /tmp/ https://storage.googleapis.com/oorb-data/oorb_data.tar.gz \
    && mkdir -p /tmp/oorb/ \
    && tar -xvf /tmp/oorb_data.tar.gz -C /tmp/oorb/
ENV OORB_DATA=/tmp/oorb/data/

# Install THOR
RUN mkdir -p /code/
WORKDIR /code/
ADD . /code/
RUN pip install -e .[dev]
