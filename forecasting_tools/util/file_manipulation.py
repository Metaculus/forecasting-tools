import datetime as dat
import functools
import json
import os
from pathlib import Path
from typing import Callable

from PIL import Image


def get_absolute_path(path_in_package: str) -> str:
    """
    This function returns the absolute path of a file in the package
    If there is no parameter given, it will just give the absolute path of the package
    @param path_in_package: The path of the file in the package starting just after the package name (e.g. "data/claims.csv")
    """
    # If it's already an absolute path, return it as is
    if os.path.isabs(path_in_package):
        return path_in_package

    path_in_package = (
        os.path.normpath(path_in_package.strip("/"))
        if path_in_package != ""
        else ""
    )

    package_name = _get_package_name()
    package_path = _get_absolute_path_of_directory(package_name)

    if path_in_package.startswith(package_name):
        updated_path_in_package = path_in_package.removeprefix(
            package_name
        ).strip("/")
        absolute_path = os.path.join(package_path, updated_path_in_package)
    else:
        one_level_up_path = os.path.dirname(package_path)
        assert os.path.exists(
            os.path.join(one_level_up_path, "pyproject.toml")
        ), "pyproject.toml not found in parent directory"
        absolute_path = os.path.join(one_level_up_path, path_in_package)

    return absolute_path.rstrip("/")


def _get_package_name() -> str:
    current_path = Path(__file__)
    while current_path != current_path.parent:
        current_path = current_path.parent
        parent_path = current_path.parent
        if (parent_path / "pyproject.toml").exists() or (
            parent_path / "setup.py"
        ).exists():
            return current_path.name
    raise RuntimeError("Package name not found")


def _get_absolute_path_of_directory(name_of_directory: str) -> str:
    current_file_path = os.path.abspath(__file__)
    package_path = os.path.dirname(current_file_path)
    iterations = 0
    max_iterations = 100
    while os.path.basename(package_path) != name_of_directory:
        package_path = os.path.dirname(package_path)
        iterations += 1
        if (
            iterations > max_iterations
            or package_path == "/"
            or package_path == ""
        ):
            raise RuntimeError(f"Directory {name_of_directory} not found")
    return package_path


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
    full_file_path = get_absolute_path(project_file_path)
    with open(full_file_path, "r") as file:
        return json.load(file)


def load_jsonl_file(file_path_in_package: str) -> list[dict]:
    full_file_path = get_absolute_path(file_path_in_package)
    with open(full_file_path, "r") as file:
        return [json.loads(line) for line in file]


def load_text_file(file_path_in_package: str) -> str:
    full_file_path = get_absolute_path(file_path_in_package)
    with open(full_file_path, "r") as file:
        return file.read()


@skip_if_file_writing_not_allowed
def write_json_file(file_path_in_package: str, input: list[dict]) -> None:
    json_string = json.dumps(input, indent=4)
    create_or_overwrite_file(file_path_in_package, json_string)


def add_to_jsonl_file(file_path_in_package: str, input: list[dict]) -> None:
    json_strings = [json.dumps(item) for item in input]
    jsonl_string = "\n".join(json_strings)
    create_or_append_to_file(file_path_in_package, jsonl_string)


@skip_if_file_writing_not_allowed
def create_or_overwrite_file(file_path_in_package: str, text: str) -> None:
    """
    This function writes text to a file, and creates the file if it does not exist
    """
    full_file_path = get_absolute_path(file_path_in_package)
    os.makedirs(os.path.dirname(full_file_path), exist_ok=True)
    with open(full_file_path, "w") as file:
        file.write(text)


@skip_if_file_writing_not_allowed
def create_or_append_to_file(file_path_in_package: str, text: str) -> None:
    """
    This function appends text to a file, and creates the file if it does not exist
    """
    full_file_path = get_absolute_path(file_path_in_package)
    os.makedirs(os.path.dirname(full_file_path), exist_ok=True)
    with open(full_file_path, "a") as file:
        file.write(text)


@skip_if_file_writing_not_allowed
def log_to_file(
    file_path_in_package: str, text: str, type: str = "DEBUG"
) -> None:
    """
    This function writes text to a file but adds a time stamp and a type statement
    """
    new_text = f"{type} - {dat.datetime.now()} - {text}"
    full_file_path = get_absolute_path(file_path_in_package)
    os.makedirs(os.path.dirname(full_file_path), exist_ok=True)
    with open(full_file_path, "a+") as file:
        file.write(new_text + "\n")


@skip_if_file_writing_not_allowed
def write_image_file(
    file_path_in_package: str, image: Image.Image, format: str | None = None
) -> None:
    full_file_path = get_absolute_path(file_path_in_package)
    os.makedirs(os.path.dirname(full_file_path), exist_ok=True)
    image.save(full_file_path, format=format)


def current_date_time_string() -> str:
    return dat.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == "__main__":
    """
    This is the "main" code area, and can be used for quickly sandboxing and testing functions
    """
    pass
