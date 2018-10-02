FROM ubuntu:xenial

LABEL maintainer="arjan.verkerk@nelen-schuurmans.nl"

# Create a nens user and group, with IDs matching those of the developer.
# The default values can be overridden at build-time via:
#
# docker-compose build --build-arg uid=`id -u` --build-arg gid=`id -g` lib
#
# The -l option is to fix a problem with large user IDs (e.g. 1235227245).
# https://forums.docker.com/t/run-adduser-seems-to-hang-with-large-uid/27371/3
# https://github.com/moby/moby/issues/5419
ARG uid=1000
ARG gid=1000
RUN groupadd -g $gid nens && useradd -lm -u $uid -g $gid nens

# Get rid of debconf messages like "unable to initialize frontend: Dialog".
# https://github.com/docker/docker/issues/4032#issuecomment-192327844
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install --yes \
        git \
        libgdal-dev \
        libhdf5-serial-dev \
        libnetcdf-dev \
        locales \
        python-dev \
        python-pip \
    && apt autoremove --yes \
    && apt clean

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

RUN pip install --upgrade pip pipenv setuptools

VOLUME /code
WORKDIR /code
USER nens
