import logging

logger = logging.getLogger(__name__)
import os
from typing import Any

import requests

from forecasting_tools.util.misc import raise_for_status_with_additional_info


class CodaUtils:
    CODA_API_KEY = os.getenv("CODA_API_KEY")


class CodaColumn:
    def __init__(self, column_name: str, column_id: str):
        self.column_name = column_name
        self.column_id = column_id


class CodaCell:
    def __init__(self, column: CodaColumn, value: Any):
        self.column = column
        non_none_value = "" if value is None else value
        self.value = non_none_value

    def turn_to_payload_friendly_json(self) -> dict:
        return {"column": self.column.column_id, "value": self.value}


class CodaRow:
    def __init__(self, cells: list[CodaCell]):
        self.cells = cells

    def turn_to_payload_friendly_json(self) -> list[dict]:
        """
        This function turns a CodaRow into a payload friendly json
        """
        cell_jsons: list[dict] = [
            cell.turn_to_payload_friendly_json() for cell in self.cells
        ]
        payload = [{"cells": cell_jsons}]
        return payload


class CodaTable:
    MAX_SIZE_OF_PAYLOAD_UPLOAD_IN_KB = 85

    def __init__(
        self,
        doc_id: str,
        table_id: str,
        columns: list[CodaColumn],
        key_columns: list[CodaColumn],
    ):
        self.doc_id = doc_id
        self.table_id = table_id
        self.columns = columns
        self.key_columns = key_columns

    def add_row_to_table(self, row: CodaRow):
        assert self.check_that_row_matches_columns(row), "Row does not match columns"
        json_payload = row.turn_to_payload_friendly_json()
        key_columns = [column.column_id for column in self.key_columns]
        headers = {"Authorization": f"Bearer {CodaUtils.CODA_API_KEY}"}
        uri = f"https://coda.io/apis/v1/docs/{self.doc_id}/tables/{self.table_id}/rows"
        logger.info(f"Attempting to insert {len(json_payload)} rows into")
        full_payload = {"rows": json_payload, "keyColumns": key_columns}
        response = requests.post(uri, headers=headers, json=full_payload)
        logger.info(f"Got response back - {response}")
        raise_for_status_with_additional_info(response)
        return response

    def check_that_row_matches_columns(self, row: CodaRow):
        cell_columns = [cell.column for cell in row.cells]

        for column in self.columns:
            if column not in cell_columns:
                return False

        for cell in row.cells:
            if cell.column not in self.columns:
                return False

        return True
