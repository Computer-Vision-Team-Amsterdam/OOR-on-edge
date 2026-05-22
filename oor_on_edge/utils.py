import logging
import os
import shutil
import time
from datetime import datetime
from functools import wraps
from typing import List, Optional, Tuple

from pyproj import Geod

from oor_on_edge.metadata import FrameMetadata

logger = logging.getLogger(__name__)

GEODETIC = Geod(ellps="WGS84")


class Speedometer:
    """
    Speedometer based on GPS metadata. The speed is provided as an Exponential
    Moving Average over the data points. The exponent can be controlled using
    the `ema_factor`, where bigger values average over more history yielding a
    smoother signal, and 1 means no history.

    Parameters
    ----------
    ema_factor: int = 10
        Used as: `ema = ema + (new_value - ema) / ema_factor`
    """

    last_location: Optional[Tuple[float, float]]
    last_timestamp: Optional[datetime]
    ema: float
    counter: int
    session: Optional[str]

    def __init__(self, ema_factor: int = 10):
        self.ema_factor = ema_factor
        self.reset()

    def update(self, frame_metadata: FrameMetadata) -> Tuple[float, float]:
        """
        Update the speed based on a new GPS data point. The previous datapoint
        is used to compute the distance and duration of the update, upon which
        the speed is based.

        Parameters
        ----------
        frame_metadata: :class:`~FrameMetadata`
            Metadata including the GPS coordinates and timestamp.

        Returns
        -------
        A tuple: (new EMA, instantaneous speed of the last step)
        """
        if self.session and self.session != os.path.dirname(
            frame_metadata.get_image_full_path()
        ):
            self.reset()
            return self.update(frame_metadata=frame_metadata)

        if (
            self.session is None
            or self.last_location is None
            or self.last_timestamp is None
        ):
            self.session = os.path.dirname(frame_metadata.get_image_full_path())
            self.last_location = frame_metadata.get_lat_lon()
            self.last_timestamp = frame_metadata.get_timestamp()
            return (-1, -1)

        distance = get_distance(frame_metadata.get_lat_lon(), self.last_location)
        duration = frame_metadata.get_timestamp() - self.last_timestamp
        speed = distance / duration.total_seconds()

        self.counter += 1
        self.ema = self.ema + (speed - self.ema) / min(self.counter, self.ema_factor)

        self.last_location = frame_metadata.get_lat_lon()
        self.last_timestamp = frame_metadata.get_timestamp()

        return (self.ema, speed)

    def reset(self):
        """Reset the speedometer."""
        self.last_location = None
        self.last_timestamp = None
        self.ema = 0
        self.counter = 0
        self.session = None


class MoveDetector:
    """
    MoveDetector based on GPS metadata. Movement detection is based on the
    difference in GPS coordinates since the last position where movement was
    detected.

    Parameters
    ----------
    min_dist: float = 1.0
        Minimum distance required since last stored coordinates to detect
        movement.
    timeout: float = 5.0
        Timeout since last detected movement for which the detector keeps its
        current status, used to compensate for intermittent GPS updates.
    """

    last_location: Optional[Tuple[float, float]]
    last_timestamp: Optional[datetime]
    session: Optional[str]

    def __init__(self, min_dist: float = 1.0, timeout: float = 5.0):
        self.min_dist = min_dist
        self.timeout = timeout
        self.reset()

    def update(self, frame_metadata: FrameMetadata) -> bool:
        """
        Update the move detector based on a new GPS data point. A move is
        detected if the new GPS coordinates are at least `min_dist` away from
        the previous stored coordinates OR the time passed is less than
        `timeout`. When a move is detected the new GPS coordinates and timestamp
        are stored to compare future values against.

        Parameters
        ----------
        frame_metadata: :class:`~FrameMetadata`
            Metadata including the GPS coordinates and timestamp.

        Returns
        -------
        A boolean indicating whether the device is moving.
        """
        if self.session and self.session != os.path.dirname(
            frame_metadata.get_image_full_path()
        ):
            self.reset()
            return self.update(frame_metadata=frame_metadata)

        if (
            self.session is None
            or self.last_location is None
            or self.last_timestamp is None
        ):
            self.session = os.path.dirname(frame_metadata.get_image_full_path())
            self.last_location = frame_metadata.get_lat_lon()
            self.last_timestamp = frame_metadata.get_timestamp()
            return True

        distance = get_distance(frame_metadata.get_lat_lon(), self.last_location)
        duration = frame_metadata.get_timestamp() - self.last_timestamp

        if distance >= self.min_dist:
            self.last_location = frame_metadata.get_lat_lon()
            self.last_timestamp = frame_metadata.get_timestamp()
            return True
        elif duration.total_seconds() <= self.timeout:
            return True
        else:
            return False

    def reset(self):
        """Reset the move detector."""
        self.last_location = None
        self.last_timestamp = None
        self.session = None


def get_distance(latlon1: Tuple[float, float], latlon2: Tuple[float, float]) -> float:
    """Get the distance between two LatLon points in meters."""
    return GEODETIC.inv(latlon1[1], latlon1[0], latlon2[1], latlon2[0])[2]


def get_frame_metadata_file_paths(
    root_folder: str,
    file_type: str = ".json",
    ignore_folders: List[str] = ["processed"],
) -> List[str]:
    """
    List all files with a given file_type (default: .json) in root_folder
    recursively. Optional ignore_folders will be skipped. Returns a sorted list.

    Parameters
    ----------
    root_folder : str
        Root folder
    file_type : str = ".json"
        Type of file to filter by
    ignore_folders: List[str] = ["processed"]
        List of folder names that will be skipped

    Returns
    -------
    List[str]
        Sorted list of file paths
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in ignore_folders]
        for filename in filenames:
            if filename.endswith(file_type):
                filepath = os.path.join(dirpath, filename)
                files.append(filepath)
    return sorted(files)


def count_files_in_folder_tree(
    root_folder: str, file_type: str, ignore_folders: List[str] = []
) -> int:
    """
    Counts how many files of a specific type are in a folder and all the
    subfolders. Optional ignore_folders will be skipped. The type is for
    example: "json", "jpg", ...

    Parameters
    ----------
    root_folder : str
        Root folder
    file_type : str
        Type of file to filter by
    ignore_folders: List[str] = []
        List of folder names that will be skipped

    Returns
    -------
    int
        File count
    """
    count = len(
        get_frame_metadata_file_paths(
            root_folder=root_folder, file_type=file_type, ignore_folders=ignore_folders
        )
    )
    return count


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.debug(f"Finished {func.__name__} in {duration:.4f} seconds.")
        return result

    return wrapper


def move_file(file_path: str, output_file_path: str):
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        shutil.move(file_path, output_file_path)
        logger.debug(f"{file_path} has been moved to {output_file_path}.")
    except FileNotFoundError:
        logger.error(f"{file_path} does not exist.")
    except Exception as e:
        logger.error(f"Failed to move file '{file_path}': {str(e)}")
        raise Exception(f"Failed to move file '{file_path}': {e}")


def copy_file(file_path: str, output_file_path: str):
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        shutil.copyfile(file_path, output_file_path)
        logger.debug(f"{file_path} has been moved to {output_file_path}.")
    except FileNotFoundError:
        logger.error(f"{file_path} does not exist.")
    except Exception as e:
        logger.error(f"Failed to copy file '{file_path}': {str(e)}")
        raise Exception(f"Failed to copy file '{file_path}': {e}")


def delete_file(file_path: str):
    try:
        os.remove(file_path)
        logger.debug(f"{file_path} has been deleted.")
    except FileNotFoundError:
        logger.error(f"{file_path} does not exist.")
    except Exception as e:
        logger.error(f"Failed to remove file '{file_path}': {str(e)}")
        raise Exception(f"Failed to remove file '{file_path}': {e}")
