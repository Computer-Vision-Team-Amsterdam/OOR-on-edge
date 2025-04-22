import json
import os
from datetime import datetime
from typing import Any, List, Optional


class FrameMetadata:

    IMAGE_FILE_NAME_KEY = "image_file_name"
    DETECTIONS_KEY = "detections"

    def __init__(self, json_file: str, image_root_dir: Optional[str]):
        """
        Create FrameMetadata from a given JSON file.

        Parameters
        ----------
        json_file: str
            The JSON metadata file. Is expected to contain at least:

            {
                "image_file_timestamp": timestamp,  # in iso format
                "image_file_name": str,  # rel path to image w.r.t. image_root_dir
                "gps_data": {
                    "coordinate_time_stamp": timestamp,  # in iso format
                    "latitude": float,
                    "longitude": float
                }
            }

        image_root_dir: Optional[str]
            Root dir for image rel paths in JSOn metadata. If not set, the
            folder of the JSON file will be used.
        """
        with open(json_file, "r") as f:
            json_content = json.load(f)
        self.metadata = json_content
        self.file_path = json_file
        self.json_dir = os.path.dirname(json_file)
        if image_root_dir:
            self.image_root_dir = image_root_dir
        else:
            self.image_root_dir = self.json_dir

    def add_or_update_field(self, key: str, value: Any):
        self.metadata[key] = value

    def content(self) -> dict:
        return self.metadata

    def get_gps_delay(self) -> float:
        gps_time = datetime.fromisoformat(
            self.metadata["gps_data"]["coordinate_time_stamp"]
        )
        image_time = datetime.fromisoformat(self.metadata["image_file_timestamp"])
        return abs((image_time - gps_time).total_seconds())

    def gps_is_valid(self) -> bool:
        return (
            self.metadata["gps_data"]["latitude"] != 0
            and self.metadata["gps_data"]["longitude"] != 0
        )

    def get_timestamp(self) -> datetime:
        return datetime.fromisoformat(self.metadata["image_file_timestamp"])

    def get_image_filename(self) -> str:
        return os.path.basename(self.metadata[self.IMAGE_FILE_NAME_KEY])

    def get_image_full_path(self) -> str:
        return os.path.join(
            self.image_root_dir, self.metadata[self.IMAGE_FILE_NAME_KEY]
        )

    def get_file_path(self) -> str:
        return self.file_path

    def get_image_rel_path(self, image_root: Optional[str] = None) -> str:
        if not image_root:
            image_root = self.image_root_dir
        return os.path.relpath(self.get_image_full_path(), image_root)

    def get_json_rel_path(self, json_root: Optional[str] = None) -> str:
        if not json_root:
            json_root = self.json_dir
        return os.path.relpath(self.file_path, json_root)


class MetadataAggregator:

    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        self.reset()

    def append(self, frame_metadata: FrameMetadata) -> None:
        if not self.timestamp_start:
            self.timestamp_start = frame_metadata.get_timestamp()
        self.timestamp_end = frame_metadata.get_timestamp()

        self.frame_metadata_list.append(frame_metadata.content())

    def reset(self):
        self.frame_metadata_list: List[dict] = []
        self.timestamp_start = None
        self.timestamp_end = None

    def save_and_reset(self) -> None:
        if len(self.frame_metadata_list) == 0:
            self.reset()
            return

        os.makedirs(self.output_folder, exist_ok=True)
        out_file = os.path.join(
            self.output_folder,
            "raw_metadata_"
            + self.timestamp_start.strftime(format="%y%m%d_%H%M%S")
            + ".json",
        )
        json_content = {
            "timestamp_start": str(self.timestamp_start),
            "timestamp_end": str(self.timestamp_end),
            "frames": self.frame_metadata_list,
        }
        with open(out_file, "w") as f:
            json.dump(json_content, f)

        self.reset()


def get_timestamp_from_metadata_file(metadata_file: str) -> datetime:
    with open(metadata_file, "r") as f:
        json_content = json.load(f)

    if os.path.basename(metadata_file).startswith("raw_metadata"):
        return datetime.fromisoformat(json_content["timestamp_start"])
    else:
        return datetime.fromisoformat(json_content["image_file_timestamp"])


def get_img_name_from_frame_metadata(metadata_file: str) -> str:
    with open(metadata_file, "r") as f:
        json_content = json.load(f)
    return os.path.basename(json_content["image_file_name"])
