FROM ubuntu:noble

LABEL maintainer="arjan.verkerk@nelen-schuurmans.nl"

RUN userdel -r ubuntu
ARG uid=1000
ARG gid=1000
RUN groupadd -g $gid nens && useradd -lm -u $uid -g $gid nens

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
USER nens

COPY --chown=nens . .
RUN echo "export PATH=$PATH:~/.local/bin" >> ~/.bashrc
RUN pipx install --system-site-packages --editable .
