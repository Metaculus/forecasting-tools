from __future__ import annotations

import datetime
import logging
import os
import re
import sys
from logging import FileHandler, StreamHandler
from logging.handlers import RotatingFileHandler

from forecasting_tools.util import file_manipulation


# Define the filter class
class HttpApiKeyFilter(logging.Filter):
    """A filter to prevent logging API keys in HTTP requests."""

    # Compile regex for efficiency - matches http(s) URLs containing 'key='
    API_KEY_PATTERN = re.compile(r"https?://.*key=")

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filters out log records from httpx containing API keys in URLs.

        Returns False if the record should be dropped, True otherwise.
        """
        # Check if the log record is from httpx
        if record.name == "httpx":
            message = record.getMessage()
            # Check if the pattern is present in the log message
            if self.API_KEY_PATTERN.search(message):
                return False  # Discard the log record
        return True  # Keep the log record


class CustomLogger:
    _initialized = False
    DEFAULT_MESSAGE_FORMAT = (
        "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s  - %(message)s"
    )
    LATEST_DEBUG_LOG_FILE_PATH = file_manipulation.normalize_package_path(
        "logs/latest_debug.log"
    )
    ERROR_LOG_FILE_PATH = file_manipulation.normalize_package_path(
        "logs/warnings/warnings.log"
    )
    DEBUG_LOG_FILE_PATH = file_manipulation.normalize_package_path(
        "logs/debug/debug.log"
    )
    INFO_LOG_FILE_PATH = file_manipulation.normalize_package_path(
        "logs/info/info.log"
    )
    LATEST_INFO_LOG_FILE_PATH = file_manipulation.normalize_package_path(
        "logs/latest_info.log"
    )
    __message_to_append_to_file = "Message to be set..."

    @classmethod
    def setup_logging(cls) -> None:
        if cls._initialized:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Prevent specific noisy logs from propagating to root logger
        watchdog_logger = logging.getLogger(
            "watchdog.observers.inotify_buffer"
        )
        watchdog_logger.setLevel(logging.WARNING)
        watchdog_logger.propagate = False

        litellm_logger = logging.getLogger("LiteLLM")
        litellm_logger.setLevel(logging.WARNING)
        litellm_logger.propagate = False

        # Instantiate the filter to remove API key logs
        api_key_filter = HttpApiKeyFilter()

        handlers = []

        file_writing_is_allowed = (
            os.environ.get("FILE_WRITING_ALLOWED", "FALSE").upper() == "TRUE"
        )
        if file_writing_is_allowed:
            cls.__message_to_append_to_file = (
                f"Root Logger initialized at {datetime.datetime.now()}\n"
            )
            handler_1 = cls.create_persistent_log_file_handler(
                logging.WARNING, cls.ERROR_LOG_FILE_PATH
            )
            handler_2 = cls.create_persistent_log_file_handler(
                logging.INFO, cls.INFO_LOG_FILE_PATH
            )
            handler_3 = cls.create_persistent_log_file_handler(
                logging.DEBUG, cls.DEBUG_LOG_FILE_PATH
            )
            handler_4 = cls.create_latest_log_file_handler(
                logging.DEBUG, cls.LATEST_DEBUG_LOG_FILE_PATH
            )
            handler_5 = cls.create_latest_log_file_handler(
                logging.INFO, cls.LATEST_INFO_LOG_FILE_PATH
            )
            handlers.extend(
                [handler_1, handler_2, handler_3, handler_4, handler_5]
            )

        handler_6 = cls.create_stream_handler(logging.INFO)
        handlers.append(handler_6)

        for handler in handlers:
            handler.addFilter(api_key_filter)  # Add the filter to the handler
            root_logger.addHandler(handler)

        cls.clear_latest_log_files()
        root_logger.info(
            f"Logger initialized with {len(handlers)} handlers at {datetime.datetime.now()}"
        )
        root_logger.info(
            f"Attached {api_key_filter.__class__.__name__} to all handlers."
        )

        cls._initialized = True

    @classmethod
    def create_persistent_log_file_handler(
        cls, log_level: int, file_path: str
    ) -> RotatingFileHandler:
        file_manipulation.create_or_append_to_file(
            file_path, cls.__message_to_append_to_file
        )
        formatter = logging.Formatter(cls.DEFAULT_MESSAGE_FORMAT)
        handler = RotatingFileHandler(
            file_path, maxBytes=1000000, backupCount=10
        )
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        return handler

    @classmethod
    def create_latest_log_file_handler(
        cls, log_level: int, file_path: str
    ) -> FileHandler:
        file_manipulation.create_or_append_to_file(
            file_path, cls.__message_to_append_to_file
        )
        formatter = logging.Formatter(cls.DEFAULT_MESSAGE_FORMAT)
        handler = FileHandler(file_path)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        return handler

    @classmethod
    def create_stream_handler(cls, log_level: int) -> StreamHandler:
        formatter = logging.Formatter(cls.DEFAULT_MESSAGE_FORMAT)
        handler = StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        return handler

    @classmethod
    def clear_latest_log_files(cls) -> None:
        file_manipulation.create_or_overwrite_file(
            cls.LATEST_DEBUG_LOG_FILE_PATH, ""
        )
        file_manipulation.create_or_overwrite_file(
            cls.LATEST_INFO_LOG_FILE_PATH, ""
        )
