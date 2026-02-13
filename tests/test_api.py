"""
Unit tests for REST API.

Tests FastAPI endpoints for workspace management, execution,
and API functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from ds_sandbox.api.rest import create_app


class TestAPIInit:
    """Tests for API initialization."""

    def test_create_app(self):
        """Test creating FastAPI application."""
        app = create_app()

        assert app is not None
        assert app.title == "ds-sandbox API"
        assert app.version == "1.0.0"

    def test_app_has_routes(self):
        """Test that app has expected routes."""
        app = create_app()

        routes = [r.path for r in app.routes]
        assert "/v1/health" in routes
        assert "/v1/workspaces" in routes


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_info(self):
        """Test root endpoint returns API info."""
        client = TestClient(create_app())
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "ds-sandbox API"
        assert data["version"] == "1.0.0"


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_returns_status(self):
        """Test health endpoint returns status."""
        client = TestClient(create_app())
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_health_includes_backends(self):
        """Test health includes backend information."""
        client = TestClient(create_app())
        response = client.get("/v1/health")

        data = response.json()
        assert "backends" in data
        assert isinstance(data["backends"], dict)


class TestRequestModels:
    """Tests for API request models."""

    def test_create_workspace_request(self):
        """Test CreateWorkspaceRequest model."""
        from ds_sandbox.api.rest import CreateWorkspaceRequest

        # Valid request
        request = CreateWorkspaceRequest(
            workspace_id="test-ws",
            setup_dirs=["data", "models"],
        )
        assert request.workspace_id == "test-ws"
        assert "data" in request.setup_dirs

    def test_prepare_datasets_request(self):
        """Test PrepareDatasetsRequest model."""
        from ds_sandbox.api.rest import PrepareDatasetsRequest

        request = PrepareDatasetsRequest(
            datasets=["dataset-1", "dataset-2"],
            strategy="copy",
        )
        assert len(request.datasets) == 2
        assert request.strategy == "copy"


class TestResponseModels:
    """Tests for API response models."""

    def test_health_status_model(self):
        """Test HealthStatus response model."""
        from ds_sandbox.api.rest import HealthStatus

        status = HealthStatus(
            status="healthy",
            version="1.0.0",
            backends={"docker": {"status": "healthy"}},
        )

        assert status.status == "healthy"
        assert status.version == "1.0.0"
        assert "docker" in status.backends

    def test_execution_info_model(self):
        """Test ExecutionInfo response model."""
        from ds_sandbox.api.rest import ExecutionInfo

        info = ExecutionInfo(
            execution_id="exec-123",
            workspace_id="test-ws",
            status="running",
        )

        assert info.execution_id == "exec-123"
        assert info.status == "running"


class TestMiddleware:
    """Tests for API middleware."""

    def test_request_id_added(self):
        """Test that request ID is added to response headers."""
        client = TestClient(create_app())
        response = client.get("/v1/health")

        assert "X-Request-ID" in response.headers
