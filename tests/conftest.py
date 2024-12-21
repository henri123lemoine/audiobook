from pathlib import Path

import pytest

from src.logging import LogConfig


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def log_config(temp_log_dir: Path) -> LogConfig:
    """Provide a LogConfig instance with test settings."""
    return LogConfig(
        debug=True, log_dir=temp_log_dir, app_name="test_app", retention="1 day", rotation="1 MB"
    )
