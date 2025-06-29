import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        "debug": True,
        "verbose": False,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "output_dir": "/tmp/test_output",
        "model_type": "test_model",
    }


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Provide sample data for testing."""
    return {
        "input": [1, 2, 3, 4, 5],
        "output": [2, 4, 6, 8, 10],
        "metadata": {
            "version": "1.0",
            "created": "2025-01-01",
            "author": "test",
        }
    }


@pytest.fixture
def mock_json_file(temp_dir: Path, sample_data: Dict[str, Any]) -> Path:
    """Create a mock JSON file for testing."""
    json_path = temp_dir / "test_data.json"
    with open(json_path, "w") as f:
        json.dump(sample_data, f)
    return json_path


@pytest.fixture
def mock_text_file(temp_dir: Path) -> Path:
    """Create a mock text file for testing."""
    text_path = temp_dir / "test_file.txt"
    text_path.write_text("This is a test file.\nWith multiple lines.\n")
    return text_path


@pytest.fixture
def mock_python_file(temp_dir: Path) -> Path:
    """Create a mock Python file for testing."""
    py_path = temp_dir / "test_module.py"
    py_path.write_text("""
def test_function(x: int) -> int:
    return x * 2

class TestClass:
    def __init__(self, value: int):
        self.value = value
    
    def method(self) -> int:
        return self.value + 10
""")
    return py_path


@pytest.fixture
def env_vars(monkeypatch) -> Dict[str, str]:
    """Set and manage environment variables for testing."""
    test_vars = {
        "TEST_ENV": "testing",
        "DEBUG": "1",
        "API_KEY": "test_key_123",
    }
    
    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)
    
    return test_vars


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically clean up any test files created during tests."""
    yield
    # Clean up patterns
    patterns = ["test_*.tmp", "*.test", ".test_*"]
    for pattern in patterns:
        for file in Path.cwd().glob(pattern):
            if file.is_file():
                file.unlink(missing_ok=True)
            elif file.is_dir():
                shutil.rmtree(file, ignore_errors=True)


@pytest.fixture
def mock_image_file(temp_dir: Path) -> Path:
    """Create a mock image file path for testing."""
    # Note: This creates a dummy file, not an actual image
    img_path = temp_dir / "test_image.jpg"
    img_path.write_bytes(b"dummy image data")
    return img_path


@pytest.fixture
def capture_logs(caplog):
    """Fixture to help capture and assert on log messages."""
    with caplog.at_level("DEBUG"):
        yield caplog


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers documentation
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Auto-mark tests based on their location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)