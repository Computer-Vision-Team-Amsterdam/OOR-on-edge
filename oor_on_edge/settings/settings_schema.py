from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from oor_on_edge import __version__


class SettingsSpecModel(BaseModel):
    class Config:
        extra = "forbid"


class AzureIoTSpec(SettingsSpecModel):
    hostname: str
    device_id: str
    shared_access_key: str


class DataDeliveryPipelineSpec(SettingsSpecModel):
    sleep_time: int


class InferenceModelParameters(SettingsSpecModel):
    img_size: int = 640
    conf: float = 0.5
    save_img_flag: bool = False
    save_txt_flag: bool = False
    save_conf_flag: bool = False


class DefisheyeParameters(SettingsSpecModel):
    camera_matrix: List[List[float]]
    distortion_params: List[List[float]]
    input_image_size: Tuple[int, int]


class DetectionPipelineSpec(SettingsSpecModel):
    input_path: str
    input_path_on_host: str
    metadata_rel_path: str
    quarantine_rel_path: str
    detections_output_path: str
    model_name: str
    pretrained_model_path: str
    inference_params: InferenceModelParameters
    defisheye_flag: bool
    defisheye_params: DefisheyeParameters
    target_classes: List[int]
    sensitive_classes: List[int]
    output_image_size: Optional[Tuple[int, int]] = None
    target_classes_conf: Optional[float] = None
    sensitive_classes_conf: Optional[float] = None
    draw_bounding_boxes: bool = True
    skip_invalid_gps: bool = False
    acceptable_gps_delay: float = float("inf")
    sleep_time: int
    training_mode: bool
    training_mode_destination_path: str
    save_blurred_labels: bool = False


class LoggingSpec(SettingsSpecModel):
    loglevel_own: str = "INFO"
    own_packages: List[str] = [
        "__main__",
        "oor_on_edge",
    ]
    extra_loglevels: Dict[str, str] = {}
    basic_config: Dict[str, Any] = {
        "level": "WARNING",
        "format": "%(asctime)s|%(levelname)-8s|%(name)s|%(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }
    ai_instrumentation_key: str = ""
    luna_logs_dir: str = ""
    sleep_time: int = None


class OOROnEdgeSettingsSpec(SettingsSpecModel):
    class Config:
        extra = "forbid"

    customer: str
    project_version: str = __version__
    aml_model_name: str
    aml_model_version: int
    azure_iot: AzureIoTSpec
    data_delivery_pipeline: DataDeliveryPipelineSpec
    detection_pipeline: DetectionPipelineSpec
    logging: LoggingSpec = LoggingSpec()
