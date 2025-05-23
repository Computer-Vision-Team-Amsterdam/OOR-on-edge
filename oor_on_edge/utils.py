import logging
import os
import shutil
import time
from functools import wraps
from typing import List

logger = logging.getLogger(__name__)


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
        logger.debug(f"Starting {func.__name__}...")
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
        logger.info(f"{file_path} has been moved to {output_file_path}.")
    except FileNotFoundError:
        logger.error(f"{file_path} does not exist.")
    except Exception as e:
        logger.error(f"Failed to move file '{file_path}': {str(e)}")
        raise Exception(f"Failed to move file '{file_path}': {e}")


def copy_file(file_path: str, output_file_path: str):
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        shutil.copyfile(file_path, output_file_path)
        logger.info(f"{file_path} has been moved to {output_file_path}.")
    except FileNotFoundError:
        logger.error(f"{file_path} does not exist.")
    except Exception as e:
        logger.error(f"Failed to copy file '{file_path}': {str(e)}")
        raise Exception(f"Failed to copy file '{file_path}': {e}")


def delete_file(file_path: str):
    try:
        os.remove(file_path)
        logger.info(f"{file_path} has been deleted.")
    except FileNotFoundError:
        logger.error(f"{file_path} does not exist.")
    except Exception as e:
        logger.error(f"Failed to remove file '{file_path}': {str(e)}")
        raise Exception(f"Failed to remove file '{file_path}': {e}")
