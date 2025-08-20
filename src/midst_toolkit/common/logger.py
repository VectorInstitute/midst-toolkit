"""MIDST Toolkit Logger. Borrowed heavily from the Flower Labs logger."""

import logging
import sys
from io import StringIO
from logging import LogRecord
from pathlib import Path
from typing import Any, TextIO


# Create logger
LOGGER_NAME = "midst_toolkit"
TOOLKIT_LOGGER = logging.getLogger(LOGGER_NAME)
TOOLKIT_LOGGER.setLevel(logging.DEBUG)
log = TOOLKIT_LOGGER.log  # pylint: disable=invalid-name

LOG_COLORS = {
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[92m",  # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[95m",  # Magenta
    "RESET": "\033[0m",  # Reset to default
}

StreamHandler = logging.StreamHandler[Any]


class ConsoleHandler(StreamHandler):
    def __init__(
        self,
        timestamps: bool = False,
        json: bool = False,
        colored: bool = True,
        stream: TextIO | None = None,
    ) -> None:
        """
        Console handler that allows configurable formatting.

        Args:
            timestamps: Whether or not to include timestamps. Defaults to False.
            json: Whether or not to accept json. Defaults to False.
            colored: Whether or not to apply color to the logs. Defaults to True.
            stream: To initialize the underlying StreamHandler. Defaults to None.
        """
        super().__init__(stream)
        self.timestamps = timestamps
        self.json = json
        self.colored = colored

    def emit(self, record: LogRecord) -> None:
        """
        Console handler that emits the provided record.

        Args:
            record: Record to emit
        """
        if self.json:
            record.message = record.getMessage().replace("\t", "").strip()

            # Check if the message is empty
            if not record.message:
                return

        super().emit(record)

    def format(self, record: LogRecord) -> str:
        """
        Format function that adds colors to log level.

        Args:
            record: Record to have color added

        Returns:
            String with color formatting corresponding to the log.
        """
        seperator = " " * (8 - len(record.levelname))
        if self.json:
            log_fmt = "{lvl='%(levelname)s', time='%(asctime)s', msg='%(message)s'}"
        else:
            log_fmt = (
                f"{LOG_COLORS[record.levelname] if self.colored else ''}"
                f"%(levelname)s {'%(asctime)s' if self.timestamps else ''}"
                f"{LOG_COLORS['RESET'] if self.colored else ''}"
                f": {seperator} %(message)s"
            )
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def update_console_handler(
    level: int | str | None = None,
    timestamps: bool | None = None,
    colored: bool | None = None,
) -> None:
    """
    Helper function for setting the proper logging.

    Args:
        level: Level of the logger. Defaults to None.
        timestamps: Whether to include timestamps. Defaults to None.
        colored: Whether to apply color formatting. Defaults to None.
    """
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        if isinstance(handler, ConsoleHandler):
            if level is not None:
                handler.setLevel(level)
            if timestamps is not None:
                handler.timestamps = timestamps
            if colored is not None:
                handler.colored = colored


# Configure console logger
console_handler = ConsoleHandler(
    timestamps=False,
    json=False,
    colored=True,
)
console_handler.setLevel(logging.INFO)
TOOLKIT_LOGGER.addHandler(console_handler)


def configure(identifier: str, filename: str | None = None) -> None:
    """
    Configure logging to file.

    Args:
        identifier: Identifier to front the logged string
        filename: Name of the file producing the log, if desired. Defaults to None.
    """
    # Create formatter
    string_to_input = f"{identifier} | %(levelname)s %(name)s %(asctime)s "
    string_to_input += "| %(filename)s:%(lineno)d | %(message)s"
    formatter = logging.Formatter(string_to_input)

    file_path = Path(filename) if filename else None

    if file_path:
        # Create file handler and log to disk
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        TOOLKIT_LOGGER.addHandler(file_handler)


def set_logger_propagation(child_logger: logging.Logger, value: bool = True) -> logging.Logger:
    """
    Set the logger propagation attribute.

    Args:
        child_logger: Child logger object
        value: Boolean setting for propagation. If True, both parent and child logger display messages. Otherwise,
            only the child logger displays a message. This False setting prevents duplicate logs in Colab notebooks.
            Reference: https://stackoverflow.com/a/19561320. Defaults to True.

    Returns:
        Child logger object with updated propagation setting
    """
    child_logger.propagate = value
    if not child_logger.propagate:
        child_logger.log(logging.DEBUG, "Logger propagate set to False")
    return child_logger


def redirect_output(output_buffer: StringIO) -> None:
    """
    Redirect stdout and stderr to text I/O buffer.

    Args:
        output_buffer: output buffer to be directed to the I/O buffer
    """
    sys.stdout = output_buffer
    sys.stderr = output_buffer
    console_handler.stream = sys.stdout
