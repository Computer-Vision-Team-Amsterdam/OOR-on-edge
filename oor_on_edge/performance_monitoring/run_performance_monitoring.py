import logging
import os
import socket
import traceback
from datetime import datetime
from time import sleep

import psutil
import torch
from requests import get

from oor_on_edge.settings.luna_logging import setup_luna_logging
from oor_on_edge.settings.settings import OOROnEdgeSettings
from oor_on_edge.utils import count_files_in_folder_tree


def internet(
    logger: logging.Logger, host: str = "8.8.8.8", port: int = 53, timeout: int = 3
):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    connected = False
    ip = "-"

    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        connected = True
    except socket.error as ex:
        logger.error(ex)

    if connected:
        try:
            ip = get("https://api.ipify.org", timeout=timeout).content.decode("utf8")
        except ConnectionError as ex:
            logger.error(ex)

    return connected, ip


def log_status(
    logger: logging.Logger,
    metadata_folder: str,
    images_folder: str,
    detections_output_folder: str,
):
    gpu_device_name = (
        torch.cuda.get_device_name()
        if torch.cuda.is_available()
        else "GPU not available"
    )
    ram_load = psutil.virtual_memory().percent
    cpu_load = psutil.cpu_percent()
    internet_connected, ip = internet(logger)
    internet_status = ip if internet_connected else "disconnected"
    logger.info(
        f"system_status: [internet: {internet_status}, hostname: {os.uname().nodename}, cpu: {cpu_load}, ram: {ram_load}, gpu_device_name: {gpu_device_name}]"
    )
    logger.info(
        f"folder_status: ["
        f"JSONs in metadata folder: {count_files_in_folder_tree(metadata_folder, '.json', ['processed', 'quarantine'])}, "
        f"JPGs in input folder: {count_files_in_folder_tree(images_folder, '.jpg', ['screenshots', 'backup'])}, "
        f"JSONs in detections folder: {count_files_in_folder_tree(detections_output_folder, '.json', ['quarantine'])}, "
        f"JPGs in detections folder: {count_files_in_folder_tree(detections_output_folder, '.jpg')}"
        f"]"
    )


def main():
    settings = OOROnEdgeSettings.set_from_yaml("config.yml")
    logging_file_path = f"{settings['logging']['luna_logs_dir']}/performance_monitoring/{datetime.now().strftime('%y%m%d-%H%M%S')}.txt"
    setup_luna_logging(settings["logging"], logging_file_path)
    logger = logging.getLogger("performance_monitoring")
    input_folder = settings["detection_pipeline"]["input_path"]
    detections_output_folder = settings["detection_pipeline"]["detections_output_path"]
    metadata_folder = os.path.join(
        input_folder, settings["detection_pipeline"]["metadata_rel_path"]
    )
    sleep_time = (
        settings["logging"]["sleep_time"] if settings["logging"]["sleep_time"] else 30
    )

    logger.info("Performance monitor is running. It will start providing updates soon.")
    while True:
        try:
            log_status(logger, metadata_folder, input_folder, detections_output_folder)
        except Exception:
            logger.error(
                f"Exception occurred in performance monitoring pipeline: {traceback.format_exc()}"
            )
            logger.error(traceback.format_exc())
        sleep(sleep_time)


if __name__ == "__main__":
    main()
