import logging
import os

from oor_on_edge.data_delivery_pipeline.components.iot_handler import IoTHandler
from oor_on_edge.metadata import (
    get_img_name_from_frame_metadata,
    get_timestamp_from_metadata_file,
)
from oor_on_edge.settings.settings import OOROnEdgeSettings
from oor_on_edge.utils import (
    delete_file,
    get_frame_metadata_file_paths,
    log_execution_time,
)

logger = logging.getLogger("data_delivery_pipeline")


class DataDelivery:
    def __init__(self):
        """
        Pipeline that delivers detection images and metadata to the Azure IoT
        landing zone.
        """
        settings = OOROnEdgeSettings.get_settings()
        self.iot_settings = settings["azure_iot"]
        self.detections_folder = settings["detection_pipeline"][
            "detections_output_path"
        ]

    def run_pipeline(self):
        """
        Runs the data delivery pipeline:
        1. retrieve all the images and metadata that need to be delivered;
        1. deliver the data to Azure;
        1. delete the delivered data from the device.
        """
        logger.debug(f"Running delivery pipeline on {self.detections_folder}..")

        self.iot_handler = IoTHandler(
            hostname=self.iot_settings["hostname"],
            device_id=self.iot_settings["device_id"],
            shared_access_key=self.iot_settings["shared_access_key"],
        )

        metadata_file_paths = get_frame_metadata_file_paths(
            root_folder=self.detections_folder
        )
        raw_metadata_file_paths = [
            file
            for file in metadata_file_paths
            if os.path.basename(file).startswith("raw_metadata")
        ]
        detection_metadata_file_paths = [
            file
            for file in metadata_file_paths
            if not os.path.basename(file).startswith("raw_metadata")
        ]

        logger.info(
            f"Number of files to deliver: {len(raw_metadata_file_paths)} raw metadata files "
            f"and {len(detection_metadata_file_paths)} detections."
        )

        raw_metadata_success_count = 0
        detection_metadata_success_count = 0

        for raw_metadata_file in raw_metadata_file_paths:
            success = self._deliver_raw_metadata(raw_metadata_file)
            if success:
                delete_file(raw_metadata_file)
                raw_metadata_success_count += 1

        for detection_metadata_file in detection_metadata_file_paths:
            success = self._deliver_detection_data(detection_metadata_file)
            if success:
                self._delete_detection_data(detection_metadata_file)
                detection_metadata_success_count += 1

        logger.info(
            f"Successfully delivered: {raw_metadata_success_count}/{len(raw_metadata_file_paths)} raw metadata files "
            f"and {detection_metadata_success_count}/{len(detection_metadata_file_paths)} detections."
        )

    @log_execution_time
    def _deliver_raw_metadata(self, raw_metadata_file_path: str) -> bool:
        """Upload a raw_metadata file."""
        success = False
        try:
            timestamp = get_timestamp_from_metadata_file(raw_metadata_file_path)
            upload_destination_path = f"full_frame_metadata/{timestamp.strftime('%Y-%m-%d')}/{os.path.basename(raw_metadata_file_path)}"
            self.iot_handler.upload_file(
                raw_metadata_file_path, upload_destination_path
            )
            success = True
        except FileNotFoundError as e:
            logger.error(
                f"FileNotFoundError during the delivery of: {raw_metadata_file_path}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Exception during the delivery of: {raw_metadata_file_path}: {e}"
            )
        return success

    @log_execution_time
    def _deliver_detection_data(self, detection_metadata_file_path: str) -> bool:
        """Upload detection metadata and corresponding image."""
        success = False
        try:
            timestamp = get_timestamp_from_metadata_file(detection_metadata_file_path)
            detection_upload_destination_path = f"detection_metadata/{timestamp.strftime('%Y-%m-%d')}/{os.path.basename(detection_metadata_file_path)}"
            self.iot_handler.upload_file(
                detection_metadata_file_path, detection_upload_destination_path
            )

            image_name = get_img_name_from_frame_metadata(detection_metadata_file_path)
            image_full_path = os.path.join(
                os.path.dirname(detection_metadata_file_path), image_name
            )
            image_upload_destination_path = f"images/{timestamp.strftime('%Y-%m-%d')}/{os.path.basename(image_name)}"
            self.iot_handler.upload_file(image_full_path, image_upload_destination_path)

            success = True
        except FileNotFoundError as e:
            logger.error(
                f"FileNotFoundError during the delivery of: {detection_metadata_file_path}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Exception during the delivery of: {detection_metadata_file_path}: {e}"
            )
        return success

    def _delete_detection_data(self, detection_metadata_file_path: str) -> None:
        """Delete detection metadata and corresponding image."""
        image_name = get_img_name_from_frame_metadata(detection_metadata_file_path)
        image_full_path = os.path.join(
            os.path.dirname(detection_metadata_file_path), image_name
        )

        delete_file(image_full_path)
        delete_file(detection_metadata_file_path)
