import logging
from pathlib import Path

from phenotype_network_v0.logging_utils import configure_logging


def test_repeated_configuration_does_not_duplicate_handlers() -> None:
    logger = configure_logging(logger_name="phenotype_network_v0.test")
    logger = configure_logging(logger_name="phenotype_network_v0.test")
    assert len(logger.handlers) == 1


def test_logging_creates_parent_directory_and_writes_file(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "run.log"
    logger = configure_logging(log_file=log_file, logger_name="phenotype_network_v0.file_test")
    logger.info("completed")
    for handler in logger.handlers:
        handler.flush()

    content = log_file.read_text(encoding="utf-8")
    assert "INFO" in content
    assert "phenotype_network_v0.file_test" in content
    assert "completed" in content

