import json
import logging
import os
from datetime import datetime
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)


class FrameMetadata:

    IMAGE_FILE_NAME_KEY = "image_file_name"
    IMAGE_PATH_KEY = "image_path"
    DETECTIONS_KEY = "detections"

    def __init__(self, json_file: str, input_path_on_host: str, input_path_local: str):
        """
        Create FrameMetadata from a given JSON file.

        Parameters
        ----------
        json_file: str
            The JSON metadata file. Is expected to contain at least:

            {
                "image_file_timestamp": timestamp,  # in iso format
                "image_file_name": str,  # filename of the image
                "image_path": str,  # full path of the image file
                "gps_data": {
                    "coordinate_time_stamp": timestamp,  # in iso format
                    "latitude": float,
                    "longitude": float
                }
            }

        input_path_on_host: str
            This prefix path will be stripped from the image full path in the metadata and replaced by input_path_local.
        input_path_local: str
            Root dir for image paths in JSON metadata.
        """
        with open(json_file, "r") as f:
            json_content = json.load(f)
        self.metadata = json_content
        self.file_path = json_file
        self.json_dir = os.path.dirname(json_file)
        self.image_root_dir = input_path_local

        # Update image full path from host to local (e.g. within docker container)
        self.metadata[self.IMAGE_PATH_KEY] = os.path.join(
            self.image_root_dir,
            os.path.relpath(
                path=self.metadata[self.IMAGE_PATH_KEY], start=input_path_on_host
            ),
        )

    def add_or_update_field(self, key: Union[str, List[str]], value: Any):
        """
        Add or update a field in the metadata dict.

        Parameters
        ----------
        key: Union[str, List[str]]
            Key or list of keys. When a list is given, this is treated as a
            hierarchical index.
        value: Any
            Value to add
        """
        if isinstance(key, str):
            key = [key]
        dic: dict = self.metadata
        for k in key[:-1]:
            dic = dic.setdefault(k, {})
        dic[key[-1]] = value

    def content(self) -> dict:
        """Get the content of the metadata as dict."""
        return self.metadata

    def get_gps_delay(self) -> float:
        """
        Get the GPS delay (difference between image timestamp and GPS
        timestamp) in seconds.
        """
        gps_time = datetime.fromisoformat(
            self.metadata["gps_data"]["coordinate_time_stamp"]
        )
        image_time = datetime.fromisoformat(self.metadata["image_file_timestamp"])
        return abs((image_time - gps_time).total_seconds())

    def gps_is_valid(self) -> bool:
        """Check if GPS coordinates are not zero."""
        return (
            self.metadata["gps_data"]["latitude"] != 0
            and self.metadata["gps_data"]["longitude"] != 0
        )

    def get_timestamp(self) -> datetime:
        """Get image timestamp."""
        return datetime.fromisoformat(self.metadata["image_file_timestamp"])

    def get_image_filename(self) -> str:
        """Get image file name without path."""
        return os.path.basename(self.metadata[self.IMAGE_FILE_NAME_KEY])

    def get_image_full_path(self) -> str:
        """Get image full path."""
        return self.metadata[self.IMAGE_PATH_KEY]

    def get_file_path(self) -> str:
        """Get file path of original JSON metadata file."""
        return self.file_path

    def get_image_rel_path(self, image_root: Optional[str] = None) -> str:
        """
        Get image path relative to image_root. If no image_root is given, the
        image_root_dir provided on construction is used.
        """
        if not image_root:
            image_root = self.image_root_dir
        return os.path.relpath(self.get_image_full_path(), image_root)

    def get_json_rel_path(self, json_root: Optional[str] = None) -> str:
        """
        Get JSON file path relative to json_root. If no json_root is given, the
        folder containing the original JSON file is used.
        """
        if not json_root:
            json_root = self.json_dir
        return os.path.relpath(self.file_path, json_root)


class MetadataAggregator:

    frame_metadata_list: List[dict]
    data_path: str
    timestamp_start: datetime
    timestamp_end: datetime

    def __init__(self, output_folder: str):
        """
        Create an aggregator for metadata. The purpose is to collect consecutive
        frame metadata records and occasionally write them bundled into a single
        JSON file.

        Parameters
        ----------
        output_folder: str
            Where to write the aggregated metadata.
        """
        self.output_folder = output_folder
        self.reset()

    def append(self, frame_metadata: FrameMetadata) -> None:
        """
        Append a FrameMetadata instance to the aggregator.

        If the image path (excluding file name) of the record does not match the
        path of preceding images, the aggregated metadata will first be written
        to a file and the aggregator will be reset. This is to ensure metadata
        of different recording sessions is not mixed in one metadata file.
        """
        if self.data_path and self.data_path != os.path.dirname(
            frame_metadata.get_image_full_path()
        ):
            self.save_and_reset()
            self.append(frame_metadata=frame_metadata)
        else:
            if not self.timestamp_start:
                self.timestamp_start = frame_metadata.get_timestamp()
            if not self.data_path:
                self.data_path = os.path.dirname(frame_metadata.get_image_full_path())
            self.timestamp_end = frame_metadata.get_timestamp()

            self.frame_metadata_list.append(frame_metadata.content())

    def reset(self):
        """Reset the aggregator to an empty state."""
        self.frame_metadata_list = []
        self.timestamp_start = None
        self.timestamp_end = None
        self.data_path = None

    def save_and_reset(self) -> None:
        """
        Save the aggregated metadata and reset the aggregator for further use.

        The aggregated metadata will be stored as a JSON file with the name
        `raw_metadata_YYMMDD.json` and contains the following fields:

        {
            "timestamp_start": str,
            "timestamp_end": str,
            "frames": List[dict]
        }
        """
        if len(self.frame_metadata_list) == 0:
            self.reset()
            return

        os.makedirs(self.output_folder, exist_ok=True)
        out_file = os.path.join(
            self.output_folder,
            "raw_metadata_" + self.timestamp_start.strftime("%y%m%d_%H%M%S") + ".json",
        )
        json_content = {
            "timestamp_start": str(self.timestamp_start),
            "timestamp_end": str(self.timestamp_end),
            "data_path": self.data_path,
            "frames": self.frame_metadata_list,
        }
        with open(out_file, "w") as f:
            json.dump(json_content, f, indent=4)

        self.reset()

        logger.info(f"Aggregated metadata written to {out_file}")


def get_timestamp_from_metadata_file(metadata_file: str) -> datetime:
    """
    Convenience method to get the timestamp from either a frame metadata JSON
    file, or a raw_metadata file created by the MetadataAggregator.
    """
    with open(metadata_file, "r") as f:
        json_content = json.load(f)

    if os.path.basename(metadata_file).startswith("raw_metadata"):
        return datetime.fromisoformat(json_content["timestamp_start"])
    else:
        return datetime.fromisoformat(json_content["image_file_timestamp"])


def get_img_name_from_frame_metadata(metadata_file: str) -> str:
    """
    Convenience method to get the image_name from a frame metadata JSON file.
    """
    with open(metadata_file, "r") as f:
        json_content = json.load(f)
    return os.path.basename(json_content["image_file_name"])
