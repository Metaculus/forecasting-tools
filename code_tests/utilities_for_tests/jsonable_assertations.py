import os
import random
import shutil

from forecasting_tools.util.jsonable import Jsonable


def assert_reading_and_printing_from_file_works(
    jsonable_class_to_test: type[Jsonable],
    read_path: str,
    temp_write_path: str | None = None,
) -> None:
    if temp_write_path is None:
        random_number = random.randint(0, 1000000)
        temp_write_path = f"temp/temp_{random_number}.json"
    if "temp" not in temp_write_path and "tmp" not in temp_write_path:
        raise ValueError(
            f"temp_write_path must contain the word 'temp' to prevent accidental deletion of important files. temp_write_path: {temp_write_path}"
        )

    # Read the data_type from the file
    try:
        objects_from_file: list[Jsonable] = (
            jsonable_class_to_test.load_json_from_file_path(read_path)
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create a {jsonable_class_to_test} from the json file at {read_path}. Error: {e}"
        )
    assert (
        len(objects_from_file) > 0
    ), f"No {jsonable_class_to_test} objects were loaded from the json file at {read_path}"
    for object_from_file in objects_from_file:
        assert isinstance(
            object_from_file, jsonable_class_to_test
        ), f"Loaded {jsonable_class_to_test} list from the json file at {read_path} is not of type {jsonable_class_to_test}"

    # Print the data_type to a file
    try:
        jsonable_class_to_test.save_object_list_to_file_path(
            objects_from_file, temp_write_path
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to print a {jsonable_class_to_test} to the json file at {temp_write_path}. Error: {e}"
        )

    # Read the data_type from the file
    try:
        objects_from_file: list[Jsonable] = (
            jsonable_class_to_test.load_json_from_file_path(temp_write_path)
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create a {jsonable_class_to_test} from the json file after writing current data to a temp file at {temp_write_path}. Error: {e}"
        )
    for object_from_file in objects_from_file:
        assert isinstance(
            object_from_file, jsonable_class_to_test
        ), f"Loaded {jsonable_class_to_test} list from the json file at {temp_write_path} is not of type {jsonable_class_to_test}"

    # Delete the temp file
    if os.path.isdir(temp_write_path):
        shutil.rmtree(temp_write_path)
    else:
        os.remove(temp_write_path)
