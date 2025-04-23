# OOR-on-edge

This repository provides code to run a YOLO object detection model on a mobile Jetson device. The code blurs sensitive data on-edge, and transfers only images with detections to the Azure IoT landing zone.


## Installation on the Jetson device

The easiest way is to run the code in a Docker container. Follow the [instructions on how to install the NVIDIA docker runtime](https://docs.ultralytics.com/guides/docker-quickstart/#setting-up-docker-with-nvidia-support) to allow the YOLO model to run on the GPU. Then,

### 1. Clone the code

```bash
git clone git@github.com:Computer-Vision-Team-Amsterdam/OOR-on-edge.git
```

### 2. Update parameters

In the project folder (e.g. `OOR-on-edge`), make changes to the following files:

1. Check the `Dockerfile` to make sure the right base image is chosen for your specific setup.
1. Modify `config.yml` to set the parameters.
1. Update `docker-compose.yml` to point to the right paths:

    ```yaml
    volumes:
      - type: bind
        source: # add path to input folder
        target: /input
      - type: bind
        source: # add path to folder with YOLO weights .pt or .engine file
        target: /model_artifacts
      - type: bind
        source: # path where detections will be stored temporarily
        target: /detections
      - type: bind
        source: # path where logs will be stored
        target: /cvt_logs
      - type: bind
        source: # path where a copy of all input data will be stored in case training_mode is activated in the config.yml
        target: /training_mode
    ```
1. Create a file `.iot-env` with the IoT settings:

    ```bash
    HOSTNAME_IOT="XXX"
    AI_INSTRUMENTATION_KEY="YYY"
    SHARED_ACCESS_KEY_IOT="ZZZ"
    ```

### 3. Run the docker container

Move to project folder, and build and run the docker container.

```bash
cd OOR-on-edge

docker compose up -d
```

Check if everything is well by monitoring the logs:

```bash
docker compose logs -f oor-on-edge
```

The container can be stopped with

```bash
docker compose down
```


## Local installation for developing

We use UV as package manager, which can be installed using any method mentioned on [the UV webpage](https://docs.astral.sh/uv/getting-started/installation/) if needed. For example:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the code and install dependencies:

```bash
git clone git@github.com:Computer-Vision-Team-Amsterdam/OOR-on-edge.git

cd OOR-on-edge

# Create the environment locally in the folder .venv
# Although the project supports python >= 3.8, <=3.12, the specific Jetson device might require python 3.10 (or even 3.8)
uv venv --python 3.10 

# Activate the environment
source .venv/bin/activate 

# Install dependencies
uv pip install -r pyproject.toml --extra dev
```

Install pre-commit hooks. The pre-commit hooks help to ensure that all committed code is valid and consistently formatted. We use UV to manage pre-commit as well.

```bash
uv tool install pre-commit --with pre-commit-uv --force-reinstall

# Install pre-commit hooks
uv run pre-commit install

# Optional: update pre-commit hooks
uv run pre-commit autoupdate

# Run pre-commit hooks using
uv run .git/hooks/pre-commit
```
