import json
import os
from datetime import datetime
from typing import List


class FrameMetadata:

    def __init__(self, json_file: str):
        with open(json_file, "r") as f:
            json_content = json.load(f)
        self.metadata = json_content
        self.file_path = json_file
        self.base_path = os.path.dirname(json_file)

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

    def get_image_filename(self) -> str:
        return self.metadata["image_file_name"]

    def get_image_full_path(self) -> str:
        return os.path.join(self.base_path, self.get_image_filename())

    def get_base_path(self) -> str:
        return self.base_path

    def get_file_path(self) -> str:
        return self.file_path

    def get_rel_folder(self, root: str) -> str:
        return os.path.relpath(self.base_path, root)


class MetadataAggregator:

    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        self.reset()

    def append(self, frame_metadata: FrameMetadata) -> None:
        self.frame_metadata_list.append(frame_metadata.content())

    def reset(self):
        self.frame_metadata_list: List[dict] = []
        self.timestamp = datetime.now()

    def save_and_reset(self) -> None:
        os.makedirs(self.output_folder, exist_ok=True)
        out_file = os.path.join(
            self.output_folder,
            "raw_metadata_" + self.timestamp.strftime(format="%y%m%d_%H%M%S") + ".json",
        )
        json_content = {
            "timestamp": str(self.timestamp),
            "frames": self.frame_metadata_list,
        }
        with open(out_file, "w") as f:
            json.dump(json_content, f)

        self.reset()
