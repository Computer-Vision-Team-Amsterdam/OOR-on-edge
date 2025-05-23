import copy
import logging
import os
import pathlib
import time
from datetime import datetime
from json.decoder import JSONDecodeError
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
        self.quarantine_output_folder = os.path.join(
            self.input_folder, detection_settings["quarantine_rel_path"]
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

    def _instantiate_model(self, sleep_time: int) -> YOLO:
        """
        Checks if the model is available and creates the model object.

        Parameters
        ----------
        sleep_time : int
            How long to wait for model conversion if the model is an .engine
            file and is not available yet.

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

    def _load_frame_metadata(
        self, metadata_file_path: str
    ) -> Tuple[FrameMetadata, FrameMetadata]:
        """
        Load a frame metadata JSON file and return two copies:
        - frame_metadata contains the raw metadata enriched with project info
        - raw_metadata contains only the raw metadata for aggregation
        """
        settings = OOROnEdgeSettings.get_settings()

        frame_metadata = FrameMetadata(
            json_file=metadata_file_path,
            input_path_on_host=settings["detection_pipeline"]["input_path_on_host"],
            input_path_local=self.input_folder,
        )
        raw_frame_metadata = copy.deepcopy(frame_metadata)

        frame_metadata.add_or_update_field(
            "project",
            {
                "model_name": self.model_name,
                "aml_model_version": settings["aml_model_version"],
                "project_version": settings["project_version"],
                "customer": settings["customer"],
            },
        )

        return frame_metadata, raw_frame_metadata

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

    def run_pipeline(self):
        """
        Runs the detection pipeline:
        1. find the images to detect;
        2. detect objects of target class;
        3. delete the raw images.
        """
        logger.debug(
            f"Running container detection pipeline on {self.metadata_folder}.."
        )
        metadata_file_paths = utils.get_frame_metadata_file_paths(
            root_folder=self.metadata_folder,
            ignore_folders=["processed", "quarantine"],
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
            try:
                frame_metadata, raw_frame_metadata = self._load_frame_metadata(
                    metadata_file_path
                )

                success = self._process_metadata_file(frame_metadata=frame_metadata)
                if success:
                    raw_metadata_aggregator.append(frame_metadata=raw_frame_metadata)

                    if self.training_mode:
                        self._move_data(frame_metadata=frame_metadata)
                    else:
                        self._delete_data_step(frame_metadata=frame_metadata)
            except JSONDecodeError as e:
                logger.error(
                    f"Exception during the detection of: {metadata_file_path}: {e}"
                )
                self._quarantine_data(frame_metadata_path=metadata_file_path)

        logger.info(
            f"Processed {self.image_processed_count}/{len(metadata_file_paths)} images and "
            f"detected {self.target_objects_detected_count} target objects."
        )

        raw_metadata_aggregator.save_and_reset()

    @utils.log_execution_time
    def _process_metadata_file(self, frame_metadata: FrameMetadata) -> bool:
        """
        Process the image corresponding to the given FrameMetadata file. Returns
        a boolean indicating success.
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
    def _detect_and_blur_image(
        self,
        frame_metadata: FrameMetadata,
    ) -> int:
        """
        Load the image, optionally resize and de-fisheye the image, detect target
        objects and blur sensitive data. Returns the number of target objects
        detected in the image.
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
        """
        Process the model results. If any target class objects are detected,
        blur the image, optionally draw bounding boxes around target objects,
        and save image and metadata in the detections output folder.

        Returns the number of target objects detected in the image.
        """
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

    @utils.log_execution_time
    def _delete_data_step(self, frame_metadata: FrameMetadata):
        """
        Deletes the data given by the provided FrameMetadata:
        - the corresponding image;
        - the original metadata.
        """
        utils.delete_file(frame_metadata.get_image_full_path())
        utils.delete_file(frame_metadata.get_file_path())

    @utils.log_execution_time
    def _move_data(self, frame_metadata: FrameMetadata):
        """
        Moves the data given by the provided FrameMetadata to the training_mode output folder:
        - the corresponding image;
        - the original metadata.
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

    def _quarantine_data(self, frame_metadata_path: str):
        """
        Moves the data given by the provided FrameMetadata to the quarantine folder:
        - the original metadata.
        """
        metadata_rel_path = os.path.relpath(
            path=frame_metadata_path,
            start=self.metadata_folder,
        )
        metadata_destination_file_path = os.path.join(
            self.quarantine_output_folder, metadata_rel_path
        )
        utils.move_file(frame_metadata_path, metadata_destination_file_path)
