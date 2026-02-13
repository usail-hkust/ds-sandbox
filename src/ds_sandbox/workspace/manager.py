"""
Workspace management module

Workspace lifecycle and directory management.
"""

import asyncio
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..errors import (
    DatasetNotFoundError,
    WorkspaceNotFoundError,
)
from ..types import Workspace

logger = logging.getLogger(__name__)

# Default workspace subdirectories
DEFAULT_SUBDIRS = ["data", "models", "outputs"]

# Default dataset registry path
DEFAULT_DATASET_REGISTRY = "/opt/datasets"


class WorkspaceManager:
    """
    Manages workspace lifecycle including creation, dataset preparation,
    querying, and deletion. Supports concurrent operations with proper locking.
    """

    def __init__(
        self,
        base_dir: str = "/opt/workspaces",
        dataset_registry: str = DEFAULT_DATASET_REGISTRY,
    ):
        """
        Initialize the workspace manager.

        Args:
            base_dir: Base directory for workspaces (default: /opt/workspaces)
            dataset_registry: Path to the central dataset registry
        """
        self.base_dir = Path(base_dir)
        self.dataset_registry = Path(dataset_registry)
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock_task_map: Dict[str, asyncio.Task] = {}
        self._workspaces: Dict[str, Workspace] = {}

    def _get_lock(self, workspace_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific workspace."""
        if workspace_id not in self._locks:
            self._locks[workspace_id] = asyncio.Lock()
        return self._locks[workspace_id]

    async def _cleanup_lock(self, workspace_id: str) -> None:
        """Remove lock reference when no tasks are using it."""
        lock = self._locks.get(workspace_id)
        if lock and not lock.locked():
            self._locks.pop(workspace_id, None)

    async def create_workspace(
        self,
        workspace_id: str,
        setup_dirs: Optional[List[str]] = None,
    ) -> Workspace:
        """
        Create a new workspace with the specified directory structure.

        Args:
            workspace_id: Unique workspace identifier
            setup_dirs: List of subdirectories to create (default: ["data", "models", "outputs"])

        Returns:
            Workspace object with workspace details

        Raises:
            FileExistsError: If workspace already exists
        """
        subdirs = setup_dirs or DEFAULT_SUBDIRS
        workspace_path = self.base_dir / workspace_id

        async with self._get_lock(workspace_id):
            try:
                # Check if workspace already exists in memory
                if workspace_id in self._workspaces:
                    existing = self._workspaces[workspace_id]
                    if existing.status != "archived":
                        raise FileExistsError(f"Workspace '{workspace_id}' already exists")

                # Create workspace directory
                logger.info(f"Creating workspace directory: {workspace_path}")
                workspace_path.mkdir(parents=True, exist_ok=False)

                # Create subdirectories
                created_dirs = []
                for subdir in subdirs:
                    subdir_path = workspace_path / subdir
                    subdir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(subdir_path))
                    logger.debug(f"Created subdirectory: {subdir_path}")

                # Create workspace metadata
                now = datetime.now(timezone.utc).isoformat()
                workspace = Workspace(
                    workspace_id=workspace_id,
                    host_path=str(workspace_path),
                    guest_path="/workspace",
                    subdirs=subdirs,
                    status="ready",
                    created_at=now,
                    last_used_at=now,
                )

                # Store workspace
                self._workspaces[workspace_id] = workspace
                logger.info(f"Workspace '{workspace_id}' created successfully at {workspace_path}")

                return workspace

            except Exception as e:
                logger.error(f"Failed to create workspace '{workspace_id}': {e}")
                # Cleanup on failure
                if workspace_path.exists():
                    shutil.rmtree(workspace_path, ignore_errors=True)
                raise

    async def prepare_datasets(
        self,
        workspace_id: str,
        datasets: List[str],
    ) -> List[str]:
        """
        Prepare datasets in the workspace's data directory.

        Datasets are copied or linked from the central registry to the
        workspace's data directory.

        Args:
            workspace_id: Target workspace identifier
            datasets: List of dataset names to prepare

        Returns:
            List of paths to prepared datasets

        Raises:
            WorkspaceNotFoundError: If workspace doesn't exist
            DatasetNotFoundError: If a dataset doesn't exist in registry
        """
        workspace = await self.get_workspace(workspace_id)

        async with self._get_lock(workspace_id):
            data_path = Path(workspace.host_path) / "data"
            prepared_paths = []

            for dataset_name in datasets:
                source_path = self.dataset_registry / dataset_name

                if not source_path.exists():
                    logger.error(f"Dataset not found in registry: {source_path}")
                    raise DatasetNotFoundError(dataset_name)

                dest_path = data_path / dataset_name

                try:
                    if dest_path.exists():
                        logger.debug(f"Dataset already exists: {dest_path}")
                    else:
                        # Use copy2 for files, copytree for directories
                        if source_path.is_file():
                            shutil.copy2(source_path, dest_path)
                        else:
                            shutil.copytree(source_path, dest_path)

                    prepared_paths.append(str(dest_path))
                    logger.info(f"Prepared dataset '{dataset_name}' in workspace '{workspace_id}'")

                except Exception as e:
                    logger.error(f"Failed to prepare dataset '{dataset_name}': {e}")
                    raise

            # Update last used timestamp
            workspace.last_used_at = datetime.now(timezone.utc).isoformat()

            return prepared_paths

    async def get_workspace(self, workspace_id: str) -> Workspace:
        """
        Get workspace information.

        Args:
            workspace_id: Workspace identifier

        Returns:
            Workspace object with details

        Raises:
            WorkspaceNotFoundError: If workspace doesn't exist
        """
        # Check in-memory cache first
        if workspace_id in self._workspaces:
            workspace = self._workspaces[workspace_id]
            if workspace.status != "archived":
                return workspace

        # Check if directory exists on disk
        workspace_path = self.base_dir / workspace_id
        if not workspace_path.exists():
            raise WorkspaceNotFoundError(workspace_id)

        # Reconstruct workspace from disk
        subdirs = []
        if workspace_path.exists():
            for item in workspace_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    subdirs.append(item.name)

        now = datetime.now(timezone.utc).isoformat()
        workspace = Workspace(
            workspace_id=workspace_id,
            host_path=str(workspace_path),
            guest_path="/workspace",
            subdirs=subdirs or DEFAULT_SUBDIRS,
            status="ready",
            created_at=now,
            last_used_at=now,
        )
        self._workspaces[workspace_id] = workspace

        return workspace

    async def delete_workspace(self, workspace_id: str) -> bool:
        """
        Delete a workspace and all its data.

        Args:
            workspace_id: Workspace identifier

        Returns:
            True if workspace was deleted, False if it didn't exist

        Raises:
            WorkspaceNotFoundError: If workspace doesn't exist
        """
        workspace = await self.get_workspace(workspace_id)

        async with self._get_lock(workspace_id):
            try:
                workspace_path = Path(workspace.host_path)

                if not workspace_path.exists():
                    # Already deleted from disk, just cleanup memory
                    self._workspaces.pop(workspace_id, None)
                    return False

                # Remove directory and all contents
                logger.info(f"Deleting workspace: {workspace_path}")
                shutil.rmtree(workspace_path, ignore_errors=True)

                # Update status in memory
                workspace.status = "archived"

                # Remove from active workspaces
                self._workspaces.pop(workspace_id, None)

                logger.info(f"Workspace '{workspace_id}' deleted successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to delete workspace '{workspace_id}': {e}")
                raise

    async def list_workspaces(self) -> List[Workspace]:
        """
        List all active workspaces.

        Returns:
            List of Workspace objects
        """
        # Refresh from disk
        if self.base_dir.exists():
            for item in self.base_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    try:
                        await self.get_workspace(item.name)
                    except Exception:
                        pass  # Skip invalid directories

        return list(self._workspaces.values())

    async def archive_workspace(self, workspace_id: str) -> Workspace:
        """
        Archive a workspace (mark as archived without deleting data).

        Args:
            workspace_id: Workspace identifier

        Returns:
            Updated Workspace object
        """
        workspace = await self.get_workspace(workspace_id)
        workspace.status = "archived"
        logger.info(f"Workspace '{workspace_id}' archived")
        return workspace

    async def cleanup_stale_locks(self) -> int:
        """
        Clean up any stale lock references.

        Returns:
            Number of locks cleaned up
        """
        cleaned = 0
        for workspace_id in list(self._locks.keys()):
            lock = self._locks.get(workspace_id)
            if lock and not lock.locked():
                self._locks.pop(workspace_id, None)
                cleaned += 1
        return cleaned

    async def get_workspace_path(self, workspace_id: str, subdir: str = "") -> str:
        """
        Get the full path for a workspace subdirectory.

        Args:
            workspace_id: Workspace identifier
            subdir: Subdirectory name (optional)

        Returns:
            Full path string
        """
        workspace = await self.get_workspace(workspace_id)
        base_path = Path(workspace.host_path)

        if subdir:
            base_path = base_path / subdir

        return str(base_path)

    async def workspace_exists(self, workspace_id: str) -> bool:
        """
        Check if a workspace exists.

        Args:
            workspace_id: Workspace identifier

        Returns:
            True if workspace exists, False otherwise
        """
        try:
            await self.get_workspace(workspace_id)
            return True
        except WorkspaceNotFoundError:
            return False


__all__ = ["WorkspaceManager"]
