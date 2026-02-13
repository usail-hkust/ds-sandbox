"""
Unit tests for WorkspaceManager.

Tests workspace lifecycle management including creation, deletion,
dataset preparation, and workspace state management.
"""

import pytest
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, patch

from ds_sandbox.errors import WorkspaceNotFoundError, DatasetNotFoundError
from ds_sandbox.types import Workspace
from ds_sandbox.workspace.manager import WorkspaceManager


class TestWorkspaceManagerInit:
    """Tests for WorkspaceManager initialization."""

    def test_default_initialization(self, temp_dir):
        """Test WorkspaceManager with default values."""
        manager = WorkspaceManager()

        assert str(manager.base_dir) == "/opt/workspaces"
        assert str(manager.dataset_registry) == "/opt/datasets"

    def test_custom_initialization(self, workspace_base_dir, dataset_registry_dir):
        """Test WorkspaceManager with custom directories."""
        manager = WorkspaceManager(
            base_dir=str(workspace_base_dir),
            dataset_registry=str(dataset_registry_dir),
        )

        assert manager.base_dir == workspace_base_dir
        assert manager.dataset_registry == dataset_registry_dir

    def test_locks_initialized(self, temp_dir):
        """Test that locks dictionary is initialized."""
        manager = WorkspaceManager(base_dir=str(temp_dir))

        assert manager._locks == {}
        assert manager._workspaces == {}


class TestWorkspaceCreation:
    """Tests for workspace creation."""

    @pytest.mark.asyncio
    async def test_create_workspace_success(self, workspace_manager):
        """Test successful workspace creation."""
        workspace = await workspace_manager.create_workspace(
            workspace_id="test-ws-001",
            setup_dirs=["data", "models", "outputs"],
        )

        assert workspace.workspace_id == "test-ws-001"
        assert "data" in workspace.subdirs
        assert "models" in workspace.subdirs
        assert "outputs" in workspace.subdirs
        assert workspace.status == "ready"
        assert workspace.guest_path == "/workspace"

    @pytest.mark.asyncio
    async def test_create_workspace_default_subdirs(self, workspace_manager):
        """Test workspace creation with default subdirectories."""
        workspace = await workspace_manager.create_workspace(
            workspace_id="test-ws-002",
        )

        assert "data" in workspace.subdirs
        assert "models" in workspace.subdirs
        assert "outputs" in workspace.subdirs

    @pytest.mark.asyncio
    async def test_create_workspace_custom_subdirs(self, workspace_manager):
        """Test workspace creation with custom subdirs."""
        workspace = await workspace_manager.create_workspace(
            workspace_id="test-ws-003",
            setup_dirs=["data", "models", "custom"],
        )

        assert "custom" in workspace.subdirs
        assert workspace.host_path.endswith("test-ws-003")

    @pytest.mark.asyncio
    async def test_create_workspace_already_exists(self, workspace_manager):
        """Test creating workspace that already exists raises error."""
        await workspace_manager.create_workspace(
            workspace_id="test-ws-dup",
            setup_dirs=["data"],
        )

        with pytest.raises(FileExistsError, match="already exists"):
            await workspace_manager.create_workspace(
                workspace_id="test-ws-dup",
                setup_dirs=["data"],
            )

    @pytest.mark.asyncio
    async def test_create_workspace_creates_data_directory(self, workspace_manager, temp_dir):
        """Test that workspace creation creates data directory."""
        workspace = await workspace_manager.create_workspace(
            workspace_id="test-ws-data",
            setup_dirs=["data", "models", "outputs"],
        )

        workspace_path = Path(workspace.host_path)
        assert (workspace_path / "data").exists()
        assert (workspace_path / "models").exists()
        assert (workspace_path / "outputs").exists()


class TestWorkspaceRetrieval:
    """Tests for workspace retrieval."""

    @pytest.mark.asyncio
    async def test_get_workspace_success(self, workspace_manager):
        """Test successful workspace retrieval."""
        await workspace_manager.create_workspace(
            workspace_id="test-ws-get",
        )

        workspace = await workspace_manager.get_workspace("test-ws-get")

        assert workspace.workspace_id == "test-ws-get"
        assert workspace.status == "ready"

    @pytest.mark.asyncio
    async def test_get_workspace_not_found(self, workspace_manager):
        """Test getting non-existent workspace raises error."""
        with pytest.raises(WorkspaceNotFoundError):
            await workspace_manager.get_workspace("nonexistent")

    @pytest.mark.asyncio
    async def test_get_workspace_from_disk(self, workspace_manager, workspace_base_dir):
        """Test getting workspace that's on disk but not in memory."""
        # Create workspace directly on disk
        workspace_path = workspace_base_dir / "disk-ws"
        workspace_path.mkdir()
        (workspace_path / "data").mkdir()

        workspace = await workspace_manager.get_workspace("disk-ws")

        assert workspace.workspace_id == "disk-ws"
        assert workspace.status == "ready"

    @pytest.mark.asyncio
    async def test_workspace_exists_true(self, workspace_manager):
        """Test workspace_exists returns True for existing workspace."""
        await workspace_manager.create_workspace(
            workspace_id="test-ws-exists",
        )

        result = await workspace_manager.workspace_exists("test-ws-exists")

        assert result is True

    @pytest.mark.asyncio
    async def test_workspace_exists_false(self, workspace_manager):
        """Test workspace_exists returns False for non-existent workspace."""
        result = await workspace_manager.workspace_exists("nonexistent")

        assert result is False


class TestWorkspaceDeletion:
    """Tests for workspace deletion."""

    @pytest.mark.asyncio
    async def test_delete_workspace_success(self, workspace_manager, workspace_base_dir):
        """Test successful workspace deletion."""
        await workspace_manager.create_workspace(
            workspace_id="test-ws-delete",
        )

        result = await workspace_manager.delete_workspace("test-ws-delete")

        assert result is True
        assert not (workspace_base_dir / "test-ws-delete").exists()

    @pytest.mark.asyncio
    async def test_delete_workspace_not_found(self, workspace_manager):
        """Test deleting non-existent workspace raises error."""
        with pytest.raises(WorkspaceNotFoundError):
            await workspace_manager.delete_workspace("nonexistent")

    @pytest.mark.asyncio
    async def test_delete_workspace_removes_from_cache(self, workspace_manager):
        """Test that deleting workspace removes it from memory cache."""
        await workspace_manager.create_workspace(
            workspace_id="test-ws-cache",
        )

        # Verify in cache
        assert await workspace_manager.workspace_exists("test-ws-cache")

        # Delete
        await workspace_manager.delete_workspace("test-ws-cache")

        # Verify removed from cache
        assert not await workspace_manager.workspace_exists("test-ws-cache")


class TestWorkspaceListing:
    """Tests for workspace listing."""

    @pytest.mark.asyncio
    async def test_list_workspaces_empty(self, workspace_manager):
        """Test listing workspaces when none exist."""
        workspaces = await workspace_manager.list_workspaces()

        assert workspaces == []

    @pytest.mark.asyncio
    async def test_list_workspaces_multiple(self, workspace_manager):
        """Test listing multiple workspaces."""
        await workspace_manager.create_workspace("test-ws-1")
        await workspace_manager.create_workspace("test-ws-2")
        await workspace_manager.create_workspace("test-ws-3")

        workspaces = await workspace_manager.list_workspaces()

        assert len(workspaces) == 3
        workspace_ids = [ws.workspace_id for ws in workspaces]
        assert "test-ws-1" in workspace_ids
        assert "test-ws-2" in workspace_ids
        assert "test-ws-3" in workspace_ids


class TestWorkspaceArchiving:
    """Tests for workspace archiving."""

    @pytest.mark.asyncio
    async def test_archive_workspace(self, workspace_manager):
        """Test archiving a workspace."""
        await workspace_manager.create_workspace(
            workspace_id="test-ws-archive",
        )

        workspace = await workspace_manager.archive_workspace("test-ws-archive")

        assert workspace.status == "archived"


class TestWorkspacePath:
    """Tests for workspace path utilities."""

    @pytest.mark.asyncio
    async def test_get_workspace_path(self, workspace_manager):
        """Test getting workspace path."""
        await workspace_manager.create_workspace(
            workspace_id="test-ws-path",
            setup_dirs=["data", "outputs"],
        )

        path = await workspace_manager.get_workspace_path("test-ws-path")

        assert path.endswith("test-ws-path")

    @pytest.mark.asyncio
    async def test_get_workspace_path_with_subdir(self, workspace_manager):
        """Test getting workspace path with subdirectory."""
        await workspace_manager.create_workspace(
            workspace_id="test-ws-subdir",
            setup_dirs=["data"],
        )

        path = await workspace_manager.get_workspace_path("test-ws-subdir", "data")

        assert path.endswith("test-ws-subdir/data")


class TestDatasetPreparation:
    """Tests for dataset preparation in workspaces."""

    @pytest.mark.asyncio
    async def test_prepare_datasets_success(self, workspace_manager, dataset_registry_dir):
        """Test successful dataset preparation."""
        await workspace_manager.create_workspace(
            workspace_id="test-ws-data",
        )

        prepared = await workspace_manager.prepare_datasets(
            workspace_id="test-ws-data",
            datasets=["test-dataset-1"],
        )

        assert len(prepared) == 1
        assert "test-dataset-1" in prepared[0]

    @pytest.mark.asyncio
    async def test_prepare_multiple_datasets(self, workspace_manager, dataset_registry_dir):
        """Test preparing multiple datasets."""
        await workspace_manager.create_workspace(
            workspace_id="test-ws-multi",
        )

        prepared = await workspace_manager.prepare_datasets(
            workspace_id="test-ws-multi",
            datasets=["test-dataset-1", "test-dataset-2"],
        )

        assert len(prepared) == 2

    @pytest.mark.asyncio
    async def test_prepare_dataset_not_found(self, workspace_manager):
        """Test preparing non-existent dataset raises error."""
        await workspace_manager.create_workspace(
            workspace_id="test-ws-notfound",
        )

        with pytest.raises(DatasetNotFoundError):
            await workspace_manager.prepare_datasets(
                workspace_id="test-ws-notfound",
                datasets=["nonexistent-dataset"],
            )

    @pytest.mark.asyncio
    async def test_prepare_dataset_workspace_not_found(self, workspace_manager):
        """Test preparing datasets for non-existent workspace."""
        with pytest.raises(WorkspaceNotFoundError):
            await workspace_manager.prepare_datasets(
                workspace_id="nonexistent",
                datasets=["test-dataset-1"],
            )


class TestLockManagement:
    """Tests for workspace lock management."""

    @pytest.mark.asyncio
    async def test_concurrent_workspace_creation(self, workspace_manager):
        """Test concurrent workspace creation with locking."""
        import asyncio

        # Create multiple workspaces concurrently
        tasks = [
            workspace_manager.create_workspace(f"concurrent-ws-{i}")
            for i in range(5)
        ]

        workspaces = await asyncio.gather(*tasks)

        assert len(workspaces) == 5
        assert all(ws.status == "ready" for ws in workspaces)

    @pytest.mark.asyncio
    async def test_lock_cleanup(self, workspace_manager):
        """Test stale lock cleanup."""
        # Create a workspace
        await workspace_manager.create_workspace("test-ws-lock")

        # Clean up stale locks
        cleaned = await workspace_manager.cleanup_stale_locks()

        # Should have cleaned up any non-locked entries
        assert isinstance(cleaned, int)
