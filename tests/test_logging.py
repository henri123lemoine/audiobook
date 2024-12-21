from src.logging import LogConfig, setup_logging


def test_log_config_initialization(temp_log_dir):
    """Test basic initialization of LogConfig."""
    config = LogConfig(debug=True, log_dir=temp_log_dir, app_name="test_app")
    assert config.debug is True
    assert config.log_dir == temp_log_dir
    assert config.app_name == "test_app"


def test_file_handlers_creation(log_config):
    """Test that file handlers are created correctly."""
    handlers = log_config.get_file_handlers()
    assert "info_file" in handlers
    assert "error_file" in handlers
    assert "debug_file" in handlers


def test_logging_setup_creates_files(temp_log_dir):
    """Test that log files are created when logging is set up."""
    setup_logging(debug=True, log_dir=temp_log_dir, app_name="test_app")

    assert (temp_log_dir / "test_app.log").exists()
    assert (temp_log_dir / "test_app.error.log").exists()
    assert (temp_log_dir / "test_app.debug.log").exists()
    assert (temp_log_dir / "test_app.error.log").exists()
    assert (temp_log_dir / "test_app.debug.log").exists()
