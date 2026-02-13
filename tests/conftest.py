"""
Pytest configuration and fixtures for ds-sandbox tests.

This module provides shared fixtures and configuration for all tests.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (require real services)"
    )
    config.addinivalue_line(
        "markers", "api: API endpoint tests"
    )
    config.addinivalue_line(
        "markers", "docker: Docker backend tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )


# =============================================================================
# Async Support
# =============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Temporary Directories
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def workspace_base_dir(temp_dir) -> Path:
    """Create a temporary workspace base directory."""
    path = temp_dir / "workspaces"
    path.mkdir(parents=True)
    return path


@pytest.fixture
def dataset_registry_dir(temp_dir) -> Path:
    """Create a temporary dataset registry directory."""
    path = temp_dir / "datasets"
    path.mkdir(parents=True)
    # Create some test datasets
    (path / "test-dataset-1").mkdir()
    (path / "test-dataset-1" / "data.csv").write_text("col1,col2\n1,2\n3,4")
    (path / "test-dataset-2").mkdir()
    (path / "test-dataset-2" / "data.parquet").write_bytes(b"")
    return path


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_docker_client():
    """Create a mock Docker client."""
    mock = MagicMock()
    mock.version.return_value = {"Version": "25.0.0", "ApiVersion": "1.44"}
    mock.images.get.return_value = MagicMock()
    mock.containers.list.return_value = []
    mock.containers.create.return_value = MagicMock(
        id="abc123def456",
        short_id="abc123def456",
        status="running"
    )
    mock.containers.create.return_value.start = MagicMock()
    mock.containers.create.return_value.stop = AsyncMock()
    mock.containers.create.return_value.remove = AsyncMock()
    mock.containers.create.return_value.exec_run = AsyncMock(
        return_value=MagicMock(
            exit_code=0,
            output=b"test output"
        )
    )
    return mock


@pytest.fixture
def mock_backend():
    """Create a mock sandbox backend."""
    mock = MagicMock()
    mock.execute = AsyncMock()
    mock.health_check = AsyncMock(return_value={"status": "healthy"})
    mock.cleanup = AsyncMock()
    return mock


# =============================================================================
# Sandbox Manager Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def sandbox_config(workspace_base_dir, dataset_registry_dir) -> dict:
    """Create a test sandbox configuration."""
    return {
        "default_backend": "docker",
        "workspace_base_dir": str(workspace_base_dir),
        "dataset_registry_dir": str(dataset_registry_dir),
        "workspace_retention_days": 30,
        "default_dataset_strategy": "copy",
    }


@pytest_asyncio.fixture
async def sandbox_manager(sandbox_config) -> "SandboxManager":
    """Create a SandboxManager instance for testing."""
    from ds_sandbox.config import SandboxConfig
    from ds_sandbox.manager import SandboxManager

    config = SandboxConfig(
        default_backend=sandbox_config["default_backend"],
        workspace_base_dir=sandbox_config["workspace_base_dir"],
        dataset_registry_dir=sandbox_config["dataset_registry_dir"],
        workspace_retention_days=sandbox_config["workspace_retention_days"],
        default_dataset_strategy=sandbox_config["default_dataset_strategy"],
    )
    manager = SandboxManager(config=config)
    return manager


@pytest_asyncio.fixture
async def workspace_manager(workspace_base_dir, dataset_registry_dir) -> "WorkspaceManager":
    """Create a WorkspaceManager instance for testing."""
    from ds_sandbox.workspace.manager import WorkspaceManager

    manager = WorkspaceManager(
        base_dir=str(workspace_base_dir),
        dataset_registry=str(dataset_registry_dir),
    )
    return manager


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_code() -> str:
    """Sample safe Python code for testing."""
    return """
import pandas as pd
import numpy as np

# Simple data manipulation
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
result = df['a'].sum()
print(f"Result: {result}")
"""


@pytest.fixture
def sample_workspace_data(temp_dir) -> Generator[Path, None, None]:
    """Create a sample workspace with data."""
    workspace_path = temp_dir / "test-workspace"
    workspace_path.mkdir(parents=True)

    (workspace_path / "data").mkdir()
    (workspace_path / "models").mkdir()
    (workspace_path / "outputs").mkdir()

    # Create sample data file
    (workspace_path / "data" / "sample.csv").write_text("x,y\n1,2\n3,4")

    yield workspace_path


# =============================================================================
# Mock Execution Result
# =============================================================================

@pytest.fixture
def mock_execution_result():
    """Create a mock execution result."""
    from ds_sandbox.types import ExecutionResult

    return ExecutionResult(
        success=True,
        stdout="Hello, World!\n",
        stderr="",
        exit_code=0,
        duration_ms=1500,
        artifacts=["output/model.pkl"],
        execution_id="exec-abc123",
        workspace_id="test-workspace",
        backend="docker",
        isolation_level="container",
        metadata={},
    )
