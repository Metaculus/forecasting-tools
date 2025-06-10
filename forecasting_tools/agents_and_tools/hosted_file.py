from __future__ import annotations

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
    def upload_files_to_openai(
        cls, file_data: list[FileToUpload]
    ) -> list[HostedFile]:
        client = OpenAI()
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
        # Download zip file directly to memory
        response = requests.get(download_link, stream=True)
        response.raise_for_status()

        # Create BytesIO object from response content
        zip_content = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            zip_content.write(chunk)
        zip_content.seek(0)

        # Process zip file from memory
        extracted_files: list[FileToUpload] = []
        with zipfile.ZipFile(zip_content, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith("/"):  # Skip directories
                    continue

                # Extract file to memory
                file_data = BytesIO(zip_ref.read(file_info.filename))
                extracted_files.append(
                    FileToUpload(
                        file_data=file_data, file_name=file_info.filename
                    )
                )

        # Upload files to OpenAI
        return cls.upload_files_to_openai(extracted_files)
