from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from typing import IO, Literal, Union

import requests
from litellm import OpenAI
from pydantic import BaseModel

FileContent = Union[IO[bytes], bytes, PathLike[str]]


@dataclass
class FileToUpload:
    file_data: FileContent
    file_name: str | None


class HostedFile(BaseModel):
    file_id: str
    file_name: str | None
    host: Literal["openai"] = "openai"

    @classmethod
    def upload_files_to_openai(cls, file_data: list[FileToUpload]) -> list[HostedFile]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        client = OpenAI(api_key=api_key)
        hosted_files = []
        for file_to_upload in file_data:
            file = client.files.create(
                file=file_to_upload.file_data, purpose="assistants"
            )
            hosted_files.append(
                HostedFile(file_id=file.id, file_name=file_to_upload.file_name)
            )
        return hosted_files

    @classmethod
    def upload_zipped_files(cls, download_link: str) -> list[HostedFile]:
        # TODO: Handle zip bombs
        # Download zip file directly to memory
        response = requests.get(download_link, stream=True, timeout=60)
        response.raise_for_status()

        MAX_GB = 0.5
        MAX_ZIP_SIZE = MAX_GB * 1024 * 1024 * 1024  # e.g. 0.5 GB
        content_length = int(response.headers.get("Content-Length", 0))
        if content_length > MAX_ZIP_SIZE:
            raise ValueError(
                f"Zip file too large: {download_link} is {content_length / 1024 / 1024 / 1024} GB. Max is {MAX_GB} GB."
            )

        # Create BytesIO object from response content with size limit
        zip_content = BytesIO()
        downloaded_size = 0
        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > MAX_ZIP_SIZE:
                raise ValueError(f"Download exceeded maximum size of {MAX_GB} GB")
            zip_content.write(chunk)
        zip_content.seek(0)

        # Process zip file from memory
        extracted_files: list[FileToUpload] = []
        with zipfile.ZipFile(zip_content, "r") as zip_ref:
            info_list = zip_ref.infolist()
            if len(info_list) > 10:
                raise ValueError("Too many files in zip")

            for file_info in info_list:
                full_filename = file_info.filename
                if file_info.is_dir():  # More reliable directory check
                    continue

                # Extract file to memory
                file_data = BytesIO(zip_ref.read(full_filename))
                safe_filename = os.path.basename(full_filename).replace("..", "")
                safe_filename = "".join(
                    c for c in safe_filename if c.isalnum() or c in "._- "
                )
                extracted_files.append(
                    FileToUpload(file_data=file_data, file_name=safe_filename)
                )

        # Upload files to OpenAI
        return cls.upload_files_to_openai(extracted_files)
