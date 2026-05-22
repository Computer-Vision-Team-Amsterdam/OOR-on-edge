# We use the Ultralytics base image. The architecture can be swapped seamlessly,
# choose any of the following. See also:
# https://github.com/ultralytics/ultralytics/tree/main/docker

## Default image with GPU support
# FROM ultralytics/ultralytics

## CPU only
#FROM ultralytics/ultralytics:latest-cpu

## JetPack5.1.2 on Jetson Xavier NX, AGX Xavier, AGX Orin, Orin Nano and Orin NX
FROM ultralytics/ultralytics:latest-jetson-jetpack5

## JetPack6.1 on Jetson AGX Orin, Orin NX and Orin Nano Series
# FROM ultralytics/ultralytics:latest-jetson-jetpack6

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Europe/Amsterdam"

RUN apt-get update  \
    && apt-get upgrade -y --fix-missing \
    && apt-get install -y --no-install-recommends \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.local/bin:$PATH

WORKDIR /usr/src

COPY pyproject.toml .

RUN uv pip install --system -r pyproject.toml

COPY oor_on_edge oor_on_edge
COPY config.yml config.yml
COPY entrypoint.sh .

ENTRYPOINT ["/bin/bash", "/usr/src/entrypoint.sh"]