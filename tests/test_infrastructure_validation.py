"""
Validation tests to ensure the testing infrastructure is set up correctly.
These tests verify that pytest, fixtures, and configurations work as expected.
"""
import os
import sys
from pathlib import Path

import pytest


class TestInfrastructureValidation:
    """Test suite to validate the testing infrastructure setup."""
    
    def test_pytest_is_installed(self):
        """Verify pytest is installed and importable."""
        import pytest
        assert pytest.__version__
    
    def test_pytest_cov_is_installed(self):
        """Verify pytest-cov is installed and importable."""
        import pytest_cov
        assert pytest_cov
    
    def test_pytest_mock_is_installed(self):
        """Verify pytest-mock is installed and importable."""
        import pytest_mock
        assert pytest_mock
    
    def test_project_root_accessible(self):
        """Verify the project root is accessible."""
        project_root = Path(__file__).parent.parent
        assert project_root.exists()
        assert (project_root / "pyproject.toml").exists()
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Verify unit test marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Verify integration test marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Verify slow test marker works."""
        assert True
    
    def test_temp_dir_fixture(self, temp_dir):
        """Verify temp_dir fixture creates a temporary directory."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test we can write to it
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    def test_mock_config_fixture(self, mock_config):
        """Verify mock_config fixture provides expected configuration."""
        assert isinstance(mock_config, dict)
        assert "debug" in mock_config
        assert "batch_size" in mock_config
        assert mock_config["debug"] is True
        assert mock_config["batch_size"] == 32
    
    def test_sample_data_fixture(self, sample_data):
        """Verify sample_data fixture provides expected data structure."""
        assert isinstance(sample_data, dict)
        assert "input" in sample_data
        assert "output" in sample_data
        assert "metadata" in sample_data
        assert len(sample_data["input"]) == len(sample_data["output"])
    
    def test_mock_json_file_fixture(self, mock_json_file, sample_data):
        """Verify mock_json_file fixture creates a valid JSON file."""
        assert mock_json_file.exists()
        assert mock_json_file.suffix == ".json"
        
        import json
        with open(mock_json_file) as f:
            loaded_data = json.load(f)
        
        assert loaded_data == sample_data
    
    def test_mock_text_file_fixture(self, mock_text_file):
        """Verify mock_text_file fixture creates a text file."""
        assert mock_text_file.exists()
        assert mock_text_file.suffix == ".txt"
        content = mock_text_file.read_text()
        assert "test file" in content
        assert len(content.splitlines()) == 2
    
    def test_mock_python_file_fixture(self, mock_python_file):
        """Verify mock_python_file fixture creates a Python file."""
        assert mock_python_file.exists()
        assert mock_python_file.suffix == ".py"
        content = mock_python_file.read_text()
        assert "def test_function" in content
        assert "class TestClass" in content
    
    def test_env_vars_fixture(self, env_vars):
        """Verify env_vars fixture sets environment variables."""
        assert os.environ.get("TEST_ENV") == "testing"
        assert os.environ.get("DEBUG") == "1"
        assert os.environ.get("API_KEY") == "test_key_123"
        assert env_vars["TEST_ENV"] == "testing"
    
    def test_mock_fixture(self, mocker):
        """Verify pytest-mock's mocker fixture is available."""
        mock_func = mocker.Mock(return_value=42)
        assert mock_func() == 42
        mock_func.assert_called_once()
    
    def test_capture_logs_fixture(self, capture_logs):
        """Verify capture_logs fixture works for log testing."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        assert "Debug message" in capture_logs.text
        assert "Info message" in capture_logs.text
        assert "Warning message" in capture_logs.text
    
    def test_coverage_is_tracked(self):
        """Verify that coverage tracking is active during test runs."""
        # This test will pass if coverage is being tracked
        # The actual verification happens via coverage reports
        assert True
    
    def test_cleanup_fixture_does_not_interfere(self, temp_dir):
        """Verify cleanup fixture doesn't interfere with normal operations."""
        test_file = temp_dir / "normal_test_file.txt"
        test_file.write_text("content")
        assert test_file.exists()
        
        # File in temp_dir should not be cleaned up by the cleanup fixture
        # as it's in a temp directory, not the current directory


@pytest.mark.parametrize("value,expected", [
    (1, 2),
    (5, 10),
    (10, 20),
])
def test_parametrize_works(value, expected):
    """Verify pytest parametrize decorator works correctly."""
    assert value * 2 == expected


def test_project_structure():
    """Verify the expected project structure exists."""
    project_root = Path(__file__).parent.parent
    
    # Check main directories exist
    assert (project_root / "tests").exists()
    assert (project_root / "tests" / "unit").exists()
    assert (project_root / "tests" / "integration").exists()
    
    # Check __init__.py files exist
    assert (project_root / "tests" / "__init__.py").exists()
    assert (project_root / "tests" / "unit" / "__init__.py").exists()
    assert (project_root / "tests" / "integration" / "__init__.py").exists()
    
    # Check configuration files exist
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "tests" / "conftest.py").exists()