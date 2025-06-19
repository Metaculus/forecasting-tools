import csv
import datetime as dat
import functools
import json
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image


def normalize_package_path(path_in_package: str | Path) -> str:
    if isinstance(path_in_package, Path):
        path_in_package = str(path_in_package)
    return path_in_package


def skip_if_file_writing_not_allowed(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # NOSONAR
        not_allowed_to_write_to_files_string: str = os.environ.get(
            "FILE_WRITING_ALLOWED", "TRUE"
        )
        is_allowed = not_allowed_to_write_to_files_string.upper() == "TRUE"
        if is_allowed:
            return func(*args, **kwargs)
        else:
            print(
                "WARNING: Skipping file writing as it is set or defaults to FALSE"
            )
            return None

    return wrapper


def load_json_file(project_file_path: str) -> list[dict]:
    """
    This function loads a json file. Output can be dictionary or list of dictionaries (or other json objects)
    @param project_file_path: The path of the json file starting from top of package
    """
    full_file_path = normalize_package_path(project_file_path)
    with open(full_file_path, "r") as file:
        return json.load(file)


def load_jsonl_file(file_path_in_package: str) -> list[dict]:
    full_file_path = normalize_package_path(file_path_in_package)
    with open(full_file_path, "r") as file:
        json_list = []
        for line in file:
            json_list.append(json.loads(line))
        return json_list


def load_text_file(file_path_in_package: str) -> str:
    full_file_path = normalize_package_path(file_path_in_package)
    with open(full_file_path, "r") as file:
        return file.read()


@skip_if_file_writing_not_allowed
def append_to_text_file(file_path_in_package: str, text: str) -> None:
    create_or_append_to_file(file_path_in_package, text)


@skip_if_file_writing_not_allowed
def write_text_file(file_path_in_package: str, text: str) -> None:
    create_or_overwrite_file(file_path_in_package, text)


@skip_if_file_writing_not_allowed
def write_json_file(file_path_in_package: str, input: list[dict]) -> None:
    json_string = json.dumps(input, indent=4)
    create_or_overwrite_file(file_path_in_package, json_string)


@skip_if_file_writing_not_allowed
def add_to_jsonl_file(file_path_in_package: str, input: list[dict]) -> None:
    if not file_path_in_package.endswith(".jsonl"):
        raise ValueError("File path must end with .jsonl")
    json_strings = [json.dumps(item) for item in input]
    jsonl_string = "\n".join(json_strings) + "\n"
    create_or_append_to_file(file_path_in_package, jsonl_string)


@skip_if_file_writing_not_allowed
def create_or_overwrite_file(file_path_in_package: str, text: str) -> None:
    """
    This function writes text to a file, and creates the file if it does not exist
    """
    full_file_path = normalize_package_path(file_path_in_package)
    _create_directory_if_needed(full_file_path)
    with open(full_file_path, "w") as file:
        file.write(text)


@skip_if_file_writing_not_allowed
def create_or_append_to_file(file_path_in_package: str, text: str) -> None:
    """
    This function appends text to a file, and creates the file if it does not exist
    """
    full_file_path = normalize_package_path(file_path_in_package)
    _create_directory_if_needed(full_file_path)
    with open(full_file_path, "a") as file:
        file.write(text)


def _create_directory_if_needed(file_path: str) -> None:
    if "/" in file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)


@skip_if_file_writing_not_allowed
def log_to_file(
    file_path_in_package: str, text: str, type: str = "DEBUG"
) -> None:
    """
    This function writes text to a file but adds a time stamp and a type statement
    """
    new_text = f"{type} - {dat.datetime.now()} - {text}"
    full_file_path = normalize_package_path(file_path_in_package)
    _create_directory_if_needed(full_file_path)
    with open(full_file_path, "a+") as file:
        file.write(new_text + "\n")


@skip_if_file_writing_not_allowed
def write_image_file(
    file_path_in_package: str, image: Image.Image, format: str | None = None
) -> None:
    full_file_path = normalize_package_path(file_path_in_package)
    _create_directory_if_needed(full_file_path)
    image.save(full_file_path, format=format)


def current_date_time_string() -> str:
    return dat.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@skip_if_file_writing_not_allowed
def write_csv_file(
    file_path_in_package: str, data: list[dict[str, Any]]
) -> None:
    """
    Writes a list of dictionaries to a CSV file, using the keys of the first dictionary as headers.
    Validates that all dictionaries have the same keys.
    """
    full_file_path = normalize_package_path(file_path_in_package)
    _create_directory_if_needed(full_file_path)

    if not data:
        create_or_overwrite_file(file_path_in_package, "")
        return

    fieldnames = list(data[0].keys())

    # Ensure all dictionaries have the same keys
    for i, entry in enumerate(data):
        if set(entry.keys()) != set(fieldnames):
            missing_keys = set(fieldnames) - set(entry.keys())
            extra_keys = set(entry.keys()) - set(fieldnames)
            error_msg = f"Dictionary at index {i} has different keys than the first dictionary."
            if missing_keys:
                error_msg += f" Missing keys: {missing_keys}."
            if extra_keys:
                error_msg += f" Extra keys: {extra_keys}."
            raise ValueError(error_msg)

    with open(full_file_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def load_csv_file(file_path_in_package: str) -> list[dict[str, str]]:
    """
    Loads a CSV file and returns its contents as a list of dictionaries.
    Each row becomes a dictionary with the column headers as keys.
    """
    full_file_path = normalize_package_path(file_path_in_package)

    with open(full_file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


if __name__ == "__main__":
    """
    This is the "main" code area, and can be used for quickly sandboxing and testing functions
    """
    pass
