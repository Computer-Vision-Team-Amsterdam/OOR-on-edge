import logging
import time
import traceback
from datetime import datetime

from oor_on_edge.data_delivery_pipeline.components.data_delivery import DataDelivery
from oor_on_edge.settings.luna_logging import setup_luna_logging
from oor_on_edge.settings.settings import OOROnEdgeSettings


def main():
    settings = OOROnEdgeSettings.set_from_yaml("config.yml")
    logging_file_path = f"{settings['logging']['luna_logs_dir']}/data_delivery_pipeline/{datetime.now().strftime('%y%m%d-%H%M%S')}.txt"
    setup_luna_logging(settings["logging"], logging_file_path)
    logger = logging.getLogger("data_delivery_pipeline")
    data_delivery_pipeline = DataDelivery()
    logger.info(
        f"Running data delivery pipeline on {settings['detection_pipeline']['detections_output_path']}.."
    )
    while True:
        try:
            data_delivery_pipeline.run_pipeline()
        except Exception:
            logger.error(
                f"Exception occurred in data delivery pipeline: {traceback.format_exc()}"
            )
            logger.error(traceback.format_exc())
        time.sleep(settings["data_delivery_pipeline"]["sleep_time"])


if __name__ == "__main__":
    main()
