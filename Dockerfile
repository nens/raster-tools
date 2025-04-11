FROM ubuntu:noble

LABEL maintainer="arjan.verkerk@nelen-schuurmans.nl"

RUN apt update

RUN apt install --yes pipx
RUN apt install --yes unzip
RUN apt install --yes locales
RUN apt install --yes gdal-bin

RUN apt install --yes python3-gdal

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

VOLUME /code
WORKDIR /code

USER ubuntu
