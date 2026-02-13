"""
Unit tests for SandboxManager.

Tests the core manager functionality including backend registration,
workspace management, and execution orchestration.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from ds_sandbox.config import SandboxConfig
from ds_sandbox.errors import (
    BackendUnavailableError,
    InvalidRequestError,
)
from ds_sandbox.manager import SandboxManager, IsolationRouter
from ds_sandbox.types import ExecutionRequest, Workspace


class TestSandboxManagerInit:
    """Tests for SandboxManager initialization."""

    def test_default_initialization(self):
        """Test SandboxManager with default config and auto-loaded backends."""
        manager = SandboxManager()

        assert manager.config is not None
        # 后端应该被自动加载
        assert "docker" in manager._backends
        assert "local" in manager._backends
        assert manager._workspace_cache == {}

    def test_custom_initialization(self, workspace_base_dir):
        """Test SandboxManager with custom config."""
        config = SandboxConfig(
            default_backend="docker",
            workspace_base_dir=str(workspace_base_dir),
        )
        manager = SandboxManager(config=config)

        assert manager.config.default_backend == "docker"
        assert str(manager.config.workspace_base_dir) == str(workspace_base_dir)


class TestBackendRegistration:
    """Tests for backend registration."""

    def test_register_backend_success(self, workspace_base_dir):
        """Test successful backend registration."""
        from ds_sandbox.backends.base import SandboxBackend

        config = SandboxConfig(workspace_base_dir=str(workspace_base_dir))
        manager = SandboxManager(config=config)
        # Create a mock that looks like a SandboxBackend
        mock_backend = MagicMock(spec=SandboxBackend)

        manager.register_backend("test-backend", mock_backend)

        assert "test-backend" in manager._backends
        assert manager._backends["test-backend"] is mock_backend

    def test_register_backend_invalid_type(self, workspace_base_dir):
        """Test that invalid backend type raises TypeError."""
        config = SandboxConfig(workspace_base_dir=str(workspace_base_dir))
        manager = SandboxManager(config=config)

        with pytest.raises(TypeError, match="must be a SandboxBackend"):
            manager.register_backend("invalid", "not-a-backend")

    def test_get_backend_success(self, workspace_base_dir):
        """Test getting a registered backend."""
        config = SandboxConfig(workspace_base_dir=str(workspace_base_dir))
        manager = SandboxManager(config=config)
        mock_backend = MagicMock()
        manager._backends["test"] = mock_backend

        result = manager.get_backend("test")

        assert result is mock_backend

    def test_get_backend_not_found(self, workspace_base_dir):
        """Test getting an unregistered backend raises error."""
        config = SandboxConfig(workspace_base_dir=str(workspace_base_dir))
        manager = SandboxManager(config=config)

        with pytest.raises(BackendUnavailableError):
            manager.get_backend("nonexistent")


class TestIsolationRouter:
    """Tests for IsolationRouter backend selection logic."""

    def test_router_init(self, workspace_base_dir):
        """Test IsolationRouter initialization."""
        config = SandboxConfig(workspace_base_dir=str(workspace_base_dir))
        router = IsolationRouter(config)

        assert router.config is config
        assert router.HIGH_RISK_THRESHOLD == 0.7
        assert router.MEDIUM_RISK_THRESHOLD == 0.3

    def test_decide_backend_gpu_requires_secure(self):
        """Test that GPU request routes to secure backend (docker)."""
        config = SandboxConfig()
        router = IsolationRouter(config)

        request = ExecutionRequest(
            code="print('test')",
            workspace_id="test",
            enable_gpu=True,
        )

        result = router.decide_backend(request)

        # secure 模式当前使用 docker
        assert result == "docker"

    def test_decide_backend_network_requires_secure(self):
        """Test that network whitelist routes to secure backend (docker)."""
        config = SandboxConfig()
        router = IsolationRouter(config)

        request = ExecutionRequest(
            code="print('test')",
            workspace_id="test",
            network_policy="whitelist",
            network_whitelist=["api.example.com"],
        )

        result = router.decide_backend(request)

        # secure 模式当前使用 docker
        assert result == "docker"

    def test_decide_backend_fast_mode(self):
        """Test fast mode routing to docker."""
        config = SandboxConfig()
        router = IsolationRouter(config)

        request = ExecutionRequest(
            code="print('test')",
            workspace_id="test",
            mode="fast",
        )

        result = router.decide_backend(request)

        assert result == "docker"

    def test_decide_backend_safe_mode(self):
        """Test safe mode routing to docker."""
        config = SandboxConfig()
        router = IsolationRouter(config)

        request = ExecutionRequest(
            code="print('test')",
            workspace_id="test",
            mode="safe",
        )

        result = router.decide_backend(request)

        assert result == "docker"

    def test_decide_backend_secure_mode(self):
        """Test secure mode routing to firecracker."""
        config = SandboxConfig()
        router = IsolationRouter(config)

        request = ExecutionRequest(
            code="print('test')",
            workspace_id="test",
            mode="secure",
        )

        result = router.decide_backend(request)

        # secure 模式当前使用 docker
        assert result == "docker"

    def test_decide_backend_default(self):
        """Test default backend selection."""
        config = SandboxConfig(default_backend="docker")
        router = IsolationRouter(config)

        request = ExecutionRequest(
            code="print('test')",
            workspace_id="test",
        )

        result = router.decide_backend(request)

        assert result == "docker"

    def test_decide_backend_manual_local(self):
        """Test manual local backend selection via default_backend=local."""
        config = SandboxConfig(default_backend="local")
        router = IsolationRouter(config)

        request = ExecutionRequest(
            code="print('test')",
            workspace_id="test",
        )

        result = router.decide_backend(request)

        assert result == "local"

    def test_decide_backend_manual_local_gpu_fallback(self):
        """Test local backend falls back to docker when GPU is requested."""
        config = SandboxConfig(default_backend="local")
        router = IsolationRouter(config)

        request = ExecutionRequest(
            code="print('test')",
            workspace_id="test",
            enable_gpu=True,
        )

        result = router.decide_backend(request)

        assert result == "docker"

    def test_decide_backend_manual_local_network_fallback(self):
        """Test local backend falls back to docker for restricted network policies."""
        config = SandboxConfig(default_backend="local")
        router = IsolationRouter(config)

        request = ExecutionRequest(
            code="print('test')",
            workspace_id="test",
            network_policy="whitelist",
            network_whitelist=["api.example.com"],
        )

        result = router.decide_backend(request)

        assert result == "docker"


class TestRequestValidation:
    """Tests for request validation."""

    def test_validate_empty_code(self):
        """Test that empty code raises error."""
        config = SandboxConfig()
        router = IsolationRouter(config)

        request = ExecutionRequest(code="", workspace_id="test")

        with pytest.raises(InvalidRequestError, match="Code cannot be empty"):
            router._validate_request(request)


class TestGetSupportedBackends:
    """Tests for supported backends listing."""

    def test_get_supported_backends(self):
        """Test getting list of supported backends."""
        config = SandboxConfig()
        router = IsolationRouter(config)

        backends = router.get_supported_backends()

        # 当前支持 docker + local
        assert "docker" in backends
        assert "local" in backends
        assert len(backends) == len(set(backends))
