"""
Unit tests for Docker backend.

Tests Docker sandbox execution functionality with mocked Docker calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from ds_sandbox.backends.docker import DockerSandbox
from ds_sandbox.backends.base import SandboxBackend
from ds_sandbox.types import ExecutionRequest, ExecutionResult, Workspace


class TestDockerSandboxInit:
    """Tests for DockerSandbox initialization."""

    def test_default_init(self):
        """Test DockerSandbox with default config."""
        backend = DockerSandbox()

        assert backend.image == "python:3.10-slim"
        assert backend.default_memory_mb == 4096
        assert backend.default_cpu_cores == 2.0
        assert backend.default_timeout_sec == 3600
        assert backend.network_disabled is True

    def test_custom_init(self):
        """Test DockerSandbox with custom config."""
        backend = DockerSandbox(
            config={
                "image": "python:3.11-slim",
                "memory_mb": 8192,
                "cpu_cores": 4.0,
                "timeout_sec": 7200,
                "network_disabled": False,
            }
        )

        assert backend.image == "python:3.11-slim"
        assert backend.default_memory_mb == 8192
        assert backend.default_cpu_cores == 4.0
        assert backend.default_timeout_sec == 7200
        assert backend.network_disabled is False

    def test_is_sandbox_backend(self):
        """Test DockerSandbox inherits from SandboxBackend."""
        backend = DockerSandbox()

        assert isinstance(backend, SandboxBackend)


class TestDockerHealthCheck:
    """Tests for Docker health check."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when Docker is healthy."""
        backend = DockerSandbox()
        mock_client = MagicMock()
        mock_client.version.return_value = {"Version": "25.0.0", "ApiVersion": "1.44"}
        backend._client = mock_client

        result = await backend.health_check()

        assert result["status"] == "healthy"
        assert result["backend"] == "docker"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test health check when Docker is unavailable."""
        from docker.errors import DockerException

        backend = DockerSandbox()
        mock_client = MagicMock()
        mock_client.version.side_effect = DockerException("Connection refused")
        backend._client = mock_client

        result = await backend.health_check()

        assert result["status"] == "unhealthy"
        assert "error" in result


class TestDockerVolumeBindings:
    """Tests for Docker volume binding generation."""

    def test_build_volume_bindings(self):
        """Test building Docker volume bindings."""
        backend = DockerSandbox()

        bindings = backend._build_volume_bindings(
            host_path="/tmp/workspaces/test-ws",
            subdirs=["data", "models", "outputs"],
        )

        # Should be a dict
        assert isinstance(bindings, dict)
        # Should have main workspace binding
        assert any("/workspace" in k and "test-ws" in k for k in bindings)

    def test_volume_bindings_empty_subdirs(self):
        """Test building bindings with empty subdirs."""
        backend = DockerSandbox()

        bindings = backend._build_volume_bindings(
            host_path="/tmp/workspaces/test-ws",
            subdirs=[],
        )

        # Should be a dict
        assert isinstance(bindings, dict)
        # Should have main workspace binding
        assert any("/workspace" in k and "test-ws" in k for k in bindings)


class TestDockerContainerCleanup:
    """Tests for Docker container cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_container(self):
        """Test container cleanup."""
        backend = DockerSandbox()
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.stop = AsyncMock()
        mock_container.remove = AsyncMock()

        await backend._cleanup_container(mock_container)

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_not_running(self):
        """Test cleanup when container is not running."""
        backend = DockerSandbox()
        mock_container = MagicMock()
        mock_container.status = "exited"
        mock_container.stop = AsyncMock()
        mock_container.remove = AsyncMock()

        await backend._cleanup_container(mock_container)

        # Should still remove but not stop
        mock_container.stop.assert_not_called()
        mock_container.remove.assert_called_once()


class TestDockerWorkspaceCleanup:
    """Tests for workspace-level cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_workspace(self):
        """Test cleanup for specific workspace."""
        backend = DockerSandbox()
        mock_container = MagicMock()
        mock_container.short_id = "abc123def"
        mock_container.stop = AsyncMock()
        mock_container.remove = AsyncMock()
        mock_client = MagicMock()
        mock_client.containers.get.return_value = mock_container
        backend._client = mock_client

        await backend.cleanup("test-workspace-12345")

        mock_client.containers.get.assert_called_once()


class TestDockerImagePull:
    """Tests for Docker image pulling."""

    @pytest.mark.asyncio
    async def test_pull_image_if_not_exists(self):
        """Test pulling image when not present locally."""
        from docker.errors import NotFound

        backend = DockerSandbox()
        mock_client = MagicMock()
        mock_client.images.get.side_effect = NotFound("Image not found")
        mock_client.images.pull = MagicMock()
        backend._client = mock_client

        await backend._pull_image("python:3.10-slim")

        mock_client.images.pull.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_pull_if_exists(self):
        """Test skipping pull when image exists."""
        backend = DockerSandbox()
        mock_client = MagicMock()
        mock_client.images.get.return_value = MagicMock()
        backend._client = mock_client

        await backend._pull_image("python:3.10-slim")

        mock_client.images.get.assert_called_once()
        mock_client.images.pull.assert_not_called()


class TestDockerClose:
    """Tests for Docker client close."""

    def test_close_client(self):
        """Test closing Docker client connection."""
        backend = DockerSandbox()
        mock_client = MagicMock()
        backend._client = mock_client

        backend.close()

        mock_client.close.assert_called_once()
        assert backend._client is None
