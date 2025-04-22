import copy
import logging
import os
import pathlib
import time
from datetime import datetime
from typing import Tuple

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from oor_on_edge import utils
from oor_on_edge.detection_pipeline.components.input_image import InputImage
from oor_on_edge.detection_pipeline.components.model_result import ModelResult
from oor_on_edge.metadata import FrameMetadata, MetadataAggregator
from oor_on_edge.settings.settings import OOROnEdgeSettings

logger = logging.getLogger("detection_pipeline")


class DataDetection:
    def __init__(
        self,
    ):
        """
        Object that find containers in the images using a pre-trained YOLO model and blurs sensitive data.
        """
        detection_settings = OOROnEdgeSettings.get_settings()["detection_pipeline"]

        self.input_folder = detection_settings["input_path"]
        self.metadata_folder = os.path.join(
            self.input_folder, detection_settings["metadata_rel_path"]
        )
        self.detections_output_folder = detection_settings["detections_output_path"]

        self.training_mode = detection_settings["training_mode"]
        self.training_mode_destination_path = pathlib.Path(
            detection_settings["training_mode_destination_path"]
        )

        self.blurred_labels_folder = os.path.join(
            detection_settings["training_mode_destination_path"],
            f"blurring_labels_{datetime.now().strftime('%y%m%d')}",
        )
        self.save_blurred_labels = detection_settings["save_blurred_labels"]
        if self.save_blurred_labels:
            os.makedirs(self.blurred_labels_folder, exist_ok=True)

        self.defisheye_flag = detection_settings["defisheye_flag"]
        self.defisheye_params = detection_settings["defisheye_params"]

        self.output_image_size = detection_settings["output_image_size"]
        inference_params = detection_settings["inference_params"]
        self.inference_params = {
            "imgsz": inference_params.get("img_size", 640),
            "save": inference_params.get("save_img_flag", False),
            "save_txt": inference_params.get("save_txt_flag", False),
            "save_conf": inference_params.get("save_conf_flag", False),
            "conf": inference_params.get("conf", 0.25),
        }
        self.model_name = detection_settings["model_name"]
        self.pretrained_model_path: str = os.path.join(
            detection_settings["pretrained_model_path"], self.model_name
        )
        self.model = self._instantiate_model(detection_settings["sleep_time"])
        self.target_classes = detection_settings["target_classes"]
        self.sensitive_classes = detection_settings["sensitive_classes"]
        self.target_classes_conf = (
            detection_settings["target_classes_conf"]
            if detection_settings["target_classes_conf"]
            else self.inference_params["conf"]
        )
        self.sensitive_classes_conf = (
            detection_settings["sensitive_classes_conf"]
            if detection_settings["sensitive_classes_conf"]
            else self.inference_params["conf"]
        )
        self.draw_bounding_boxes = detection_settings["draw_bounding_boxes"]
        self.skip_invalid_gps = detection_settings["skip_invalid_gps"]
        self.gps_accept_delay = float(detection_settings["acceptable_gps_delay"])

        logger.info(f"Inference_params: {self.inference_params}")
        logger.info(f"Pretrained_model_path: {self.pretrained_model_path}")
        logger.info(f"Yolo model: {self.model_name}")
        logger.info(f"Project_path: {self.detections_output_folder}")

    def _instantiate_model(self, sleep_time: int):
        """
        Checks if the model is available and creates the model object.

        Parameters
        ----------
        sleep_time : int
            How long to wait if it's not available.

        Returns
        -------
        YOLO
            Model object

        Raises
        ------
        FileNotFoundError
            In case the model is not found.
        """
        if self.pretrained_model_path.endswith(".engine"):
            while not os.path.isfile(self.pretrained_model_path):
                logger.info(
                    f"Model {self.model_name} not found, waiting for model_conversion_pipeline.."
                )
                time.sleep(sleep_time)
        elif not os.path.isfile(self.pretrained_model_path):
            raise FileNotFoundError(f"Model not found: {self.pretrained_model_path}")
        return YOLO(model=self.pretrained_model_path, task="detect")

    def load_frame_metadata(self, metadata_file_path: str) -> Tuple[dict, dict]:
        frame_metadata = FrameMetadata(
            json_file=metadata_file_path, image_root_dir=self.input_folder
        )
        raw_frame_metadata = copy.deepcopy(frame_metadata)

        frame_metadata.add_or_update_field("model_name", self.model_name)

        settings = OOROnEdgeSettings.get_settings()
        frame_metadata.add_or_update_field(
            "aml_model_version", settings["aml_model_version"]
        )
        frame_metadata.add_or_update_field(
            "project_version", settings["project_version"]
        )
        frame_metadata.add_or_update_field("customer", settings["customer"])

        return frame_metadata, raw_frame_metadata

    def run_pipeline(self):
        """
        Runs the detection pipeline:
            - find the images to detect;
            - detects objects of target class;
            - deletes the raw images.
        """
        logger.debug(
            f"Running container detection pipeline on {self.metadata_folder}.."
        )
        metadata_file_paths = utils.get_frame_metadata_file_paths(
            root_folder=self.metadata_folder
        )

        logger.info(
            f"Number of metadata files in detection queue: {len(metadata_file_paths)}"
        )

        self.image_processed_count = 0
        self.target_objects_detected_count = 0

        raw_metadata_aggregator = MetadataAggregator(
            output_folder=self.detections_output_folder
        )

        for metadata_file_path in metadata_file_paths:
            frame_metadata, raw_frame_metadata = self.load_frame_metadata(
                metadata_file_path
            )

            success = self._process_metadata_file(frame_metadata=frame_metadata)
            if success:
                raw_metadata_aggregator.append(frame_metadata=raw_frame_metadata)

                if self.training_mode:
                    self._move_data(frame_metadata=frame_metadata)
                else:
                    self._delete_data_step(frame_metadata=frame_metadata)

        logger.info(
            f"Processed {self.image_processed_count}/{len(metadata_file_paths)} images and "
            f"detected {self.target_objects_detected_count} target objects."
        )

        raw_metadata_aggregator.save_and_reset()

    def _accept_gps(self, frame_metadata: FrameMetadata) -> Tuple[bool, float]:
        """
        Check whether GPS signal is valid and has an acceptable delay.
        """
        gps_valid = (not self.skip_invalid_gps) or frame_metadata.gps_is_valid()
        accept_delay = True
        gps_delay = float("nan")

        if gps_valid and (self.gps_accept_delay != float("inf")):
            gps_delay = frame_metadata.get_gps_delay()
            accept_delay = gps_delay <= self.gps_accept_delay

        return (gps_valid and accept_delay), gps_delay

    @utils.log_execution_time
    def _process_metadata_file(self, frame_metadata: FrameMetadata) -> bool:
        """
        Loops through each row of the metadata csv file, detects containers and blur each image.

        Parameters
        ----------
        metadata_file_path : str
            Metadata file path.
        """
        success = False
        try:
            metadata_file_path = frame_metadata.get_file_path()
            logger.debug(f"metadata_file_path: {metadata_file_path}")

            accept_gps, gps_delay = self._accept_gps(frame_metadata=frame_metadata)
            if not accept_gps:
                logger.debug(
                    f"No valid GPS (delay={gps_delay:.1f}s), "
                    f"skipping frame: {frame_metadata.get_image_filename()}"
                )
            else:
                if os.path.isfile(frame_metadata.get_image_full_path()):
                    self.target_objects_detected_count += self._detect_and_blur_image(
                        frame_metadata=frame_metadata,
                    )
                    self.image_processed_count += 1
                else:
                    logger.debug(
                        f"Image {frame_metadata.get_image_full_path()} not found, skipping."
                    )
            success = True
        except Exception as e:
            logger.error(
                f"Exception during the detection of: {metadata_file_path}: {e}"
            )
        return success

    @utils.log_execution_time
    def _delete_data_step(self, frame_metadata: FrameMetadata):
        """
        Deletes the data that has been processed.

        Parameters
        ----------
        metadata_csv_file_path
            Path of the CSV file containing the metadata of the pictures,
            it's used to keep track of which files needs to be deleted.
        """
        utils.delete_file(frame_metadata.get_image_full_path())
        utils.delete_file(frame_metadata.get_file_path())

    @utils.log_execution_time
    def _move_data(self, frame_metadata: FrameMetadata):
        """
        Moves the data that has been processed to a training folder.

        Parameters
        ----------
        metadata_csv_file_path
            Path of the CSV file containing the metadata of the pictures,
            it's used to keep track of which files had to be detected.
        """
        image_rel_path = frame_metadata.get_image_rel_path()
        image_destination_full_path = os.path.join(
            self.training_mode_destination_path,
            image_rel_path,
        )
        utils.move_file(
            frame_metadata.get_image_full_path(), image_destination_full_path
        )

        metadata_rel_path = frame_metadata.get_json_rel_path(
            json_root=self.input_folder
        )
        metadata_destination_file_path = os.path.join(
            self.training_mode_destination_path, metadata_rel_path
        )
        utils.move_file(frame_metadata.get_file_path(), metadata_destination_file_path)

    @utils.log_execution_time
    def _detect_and_blur_image(
        self,
        frame_metadata: FrameMetadata,
    ):
        """Loads the image, resizes it, detects containers and blur sensitive data.

        Parameters
        ----------
        image_file_name : pathlib.Path
            File name of the image
        image_full_path : pathlib.Path
            Path of the image
        csv_path : pathlib.Path
            Path where the csv metadata file is stored. For example:
            /detections/folder1/file1.csv
        detections_path : pathlib.Path
            Path of detections excluding the file. For example:
            /detections/folder1

        Returns
        -------
        int
            Count of detected target objects
        """
        logger.debug(f"Detecting and blurring: {frame_metadata.get_image_filename()}")

        image = InputImage(image_full_path=frame_metadata.get_image_full_path())
        if self.output_image_size:
            image.resize(output_image_size=self.output_image_size)
        if self.defisheye_flag:
            image.defisheye(defisheye_params=self.defisheye_params)

        self.inference_params["source"] = image.image
        detection_results = self.model(**self.inference_params)[0]
        torch.cuda.empty_cache()

        n_detections = self._process_detections_and_blur_image(
            model_results=detection_results,
            frame_metadata=frame_metadata,
        )
        return n_detections

    @utils.log_execution_time
    def _process_detections_and_blur_image(
        self,
        model_results: Results,
        frame_metadata: FrameMetadata,
    ) -> int:
        detections_output_folder = os.path.dirname(
            os.path.join(
                self.detections_output_folder,
                frame_metadata.get_image_rel_path(),
            )
        )
        output_image_file_name = frame_metadata.get_image_filename()

        model_result = ModelResult(
            model_result=model_results,
            frame_metadata=frame_metadata,
            target_classes=self.target_classes,
            sensitive_classes=self.sensitive_classes,
            target_classes_conf=self.target_classes_conf,
            sensitive_classes_conf=self.sensitive_classes_conf,
            blurred_labels_folder=self.blurred_labels_folder,
            save_blurred_labels=self.save_blurred_labels,
            draw_boxes=self.draw_bounding_boxes,
        )
        n_detections = model_result.process_detections_and_blur_sensitive_data(
            image_detection_path=detections_output_folder,
            image_file_name=output_image_file_name,
        )

        return n_detections
