import logging
from unittest.mock import Mock, call, patch

from midst_toolkit.common.logger import KeyValueLogger


def test_key_value_logger_init() -> None:
    key_value_logger = KeyValueLogger()
    assert key_value_logger.key_to_value == {}
    assert key_value_logger.key_to_count == {}
    assert key_value_logger.log_level == logging.DEBUG


def test_key_value_logger_save_entry() -> None:
    key_value_logger = KeyValueLogger()
    key_value_logger.save_entry("test_key", 1.0)
    assert key_value_logger.key_to_value["test_key"] == 1.0


def test_key_value_logger_save_entry_mean() -> None:
    key_value_logger = KeyValueLogger()
    key_value_logger.save_entry_mean("test_key", 1.0)
    assert key_value_logger.key_to_value["test_key"] == 1.0
    assert key_value_logger.key_to_count["test_key"] == 1
    key_value_logger.save_entry_mean("test_key", 2.0)
    assert key_value_logger.key_to_value["test_key"] == 1.5
    assert key_value_logger.key_to_count["test_key"] == 2


def test_key_value_logger_truncate() -> None:
    key_value_logger = KeyValueLogger()
    result = key_value_logger._truncate("test string " * 3)  # total of 36 characters
    assert result == "test string test string tes..."


@patch("midst_toolkit.common.logger.log")
def test_key_value_logger_dump(mock_log: Mock) -> None:
    key_value_logger = KeyValueLogger()
    key_value_logger.save_entry("test_key", 1.0)
    key_value_logger.save_entry("test_key_2", 0.79)
    key_value_logger.dump()

    assert mock_log.call_count == 4
    mock_log.assert_has_calls(
        [
            call(logging.DEBUG, "-------------------------"),
            call(logging.DEBUG, "| test_key   | 1        |"),
            call(logging.DEBUG, "| test_key_2 | 0.79     |"),
            call(logging.DEBUG, "-------------------------"),
        ]
    )
    mock_log.reset_mock()
    assert len(key_value_logger.key_to_value) == 0
    assert len(key_value_logger.key_to_count) == 0

    key_value_logger.save_entry("test_key", 164537)
    key_value_logger.save_entry("really_long_key_with_more_than_30_characters", 0.98765357623989)
    key_value_logger.dump()

    assert mock_log.call_count == 4
    mock_log.assert_has_calls(
        [
            call(logging.DEBUG, "---------------------------------------------"),
            call(logging.DEBUG, "| really_long_key_with_more_t... | 0.988    |"),
            call(logging.DEBUG, "| test_key                       | 1.65e+05 |"),
            call(logging.DEBUG, "---------------------------------------------"),
        ]
    )


@patch("midst_toolkit.common.logger.log")
def test_key_value_logger_dump_empty(mock_log: Mock) -> None:
    key_value_logger = KeyValueLogger()
    key_value_logger.dump()

    assert mock_log.call_count == 1
    mock_log.assert_has_calls([call(logging.DEBUG, "WARNING: tried to write empty key-value dict")])
