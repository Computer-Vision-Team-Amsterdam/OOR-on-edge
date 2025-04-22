FROM ultralytics/ultralytics:latest-jetson-jetpack5

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Europe/Amsterdam"

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

COPY oor_on_edge oor_on_edge
COPY config.yml config.yml
COPY entrypoint.sh .
COPY pyproject.toml .

RUN uv pip install --system -r pyproject.toml

ENTRYPOINT ["/bin/bash", "/usr/src/entrypoint.sh"]