"""
Workspace service layer.

Provides a validated, API-friendly facade on top of WorkspaceManager.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ds_sandbox.data.registry import DatasetRegistry
from ds_sandbox.errors import InvalidRequestError
from ds_sandbox.path_utils import validate_path_component
from ds_sandbox.types import DatasetInfo, Workspace
from ds_sandbox.workspace.manager import WorkspaceManager

SUPPORTED_FORMATS = {"csv", "parquet", "json", "excel", "feather"}
EXTENSION_TO_FORMAT = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".json": "json",
    ".xlsx": "excel",
    ".xls": "excel",
    ".feather": "feather",
    ".fst": "feather",
}


class WorkspaceService:
    """
    Application service for workspace operations.

    Responsibilities:
    - Validate user input before touching storage
    - Delegate workspace lifecycle to WorkspaceManager
    - Provide consistent DatasetInfo objects for API responses
    """

    def __init__(
        self,
        base_dir: str = "/opt/workspaces",
        dataset_registry: str = "/opt/datasets",
        workspace_manager: Optional[WorkspaceManager] = None,
        dataset_registry_client: Optional[DatasetRegistry] = None,
    ) -> None:
        self.workspace_manager = workspace_manager or WorkspaceManager(
            base_dir=base_dir,
            dataset_registry=dataset_registry,
        )
        self.dataset_registry = dataset_registry_client or DatasetRegistry(dataset_registry)

    async def create_workspace(
        self,
        workspace_id: str,
        setup_dirs: Optional[List[str]] = None,
    ) -> Workspace:
        """Create a workspace after input validation."""
        validate_path_component(workspace_id, "workspace_id")
        for subdir in setup_dirs or []:
            validate_path_component(subdir, "setup_dirs")
        return await self.workspace_manager.create_workspace(workspace_id, setup_dirs)

    async def get_workspace(self, workspace_id: str) -> Workspace:
        """Get a workspace by ID."""
        validate_path_component(workspace_id, "workspace_id")
        return await self.workspace_manager.get_workspace(workspace_id)

    async def list_workspaces(self) -> List[Workspace]:
        """List all active workspaces."""
        return await self.workspace_manager.list_workspaces()

    async def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace by ID."""
        validate_path_component(workspace_id, "workspace_id")
        return await self.workspace_manager.delete_workspace(workspace_id)

    async def prepare_datasets(
        self,
        workspace_id: str,
        datasets: List[str],
        strategy: str = "copy",
    ) -> List[str]:
        """
        Prepare datasets in workspace/data.

        This service-level method validates input and delegates to WorkspaceManager.
        """
        validate_path_component(workspace_id, "workspace_id")
        for name in datasets:
            validate_path_component(name, "dataset_name")

        if strategy not in {"copy", "link"}:
            raise InvalidRequestError(
                field="strategy",
                value=strategy,
                reason="strategy must be 'copy' or 'link'",
            )

        return await self.workspace_manager.prepare_datasets(workspace_id, datasets)

    async def list_workspace_datasets(self, workspace_id: str) -> List[DatasetInfo]:
        """List datasets currently present in workspace/data."""
        validate_path_component(workspace_id, "workspace_id")
        workspace = await self.workspace_manager.get_workspace(workspace_id)
        workspace_data_dir = Path(workspace.host_path) / "data"

        registry_map: Dict[str, DatasetInfo] = {
            ds.name: ds for ds in self.dataset_registry.list_all()
        }

        datasets: List[DatasetInfo] = []
        if not workspace_data_dir.exists():
            return datasets

        for item in sorted(workspace_data_dir.iterdir(), key=lambda p: p.name):
            if not (item.is_file() or item.is_dir()):
                continue

            size_mb = self._calculate_size_mb(item)
            registry_info = registry_map.get(item.name)

            if registry_info is not None:
                format_name = registry_info.format
                checksum = registry_info.checksum
                description = registry_info.description
                tags = registry_info.tags
                registered_at = registry_info.registered_at
            else:
                format_name = self._detect_format(item)
                checksum = self._calculate_checksum(item)
                description = None
                tags = []
                registered_at = datetime.now(timezone.utc).isoformat()

            datasets.append(
                DatasetInfo(
                    name=item.name,
                    source_path=str(item),
                    size_mb=size_mb,
                    checksum=checksum,
                    format=format_name,
                    description=description,
                    tags=tags,
                    registered_at=registered_at,
                )
            )

        return datasets

    def _detect_format(self, path: Path) -> str:
        """Detect dataset format from file extension, with safe fallback."""
        if path.is_file():
            detected = EXTENSION_TO_FORMAT.get(path.suffix.lower())
            return detected if detected in SUPPORTED_FORMATS else "json"

        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    detected = EXTENSION_TO_FORMAT.get(file_path.suffix.lower())
                    if detected in SUPPORTED_FORMATS:
                        return detected
        return "json"

    def _calculate_size_mb(self, path: Path) -> float:
        """Calculate file/directory size in megabytes."""
        if path.is_file():
            size_bytes = path.stat().st_size
        else:
            size_bytes = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return round(size_bytes / (1024 * 1024), 2)

    def _calculate_checksum(self, path: Path) -> str:
        """
        Calculate deterministic checksum.

        For directories, checksum includes relative file paths and file content.
        """
        digest = hashlib.sha256()

        if path.is_file():
            self._update_digest_with_file(digest, path)
            return digest.hexdigest()

        for file_path in sorted(path.rglob("*")):
            if not file_path.is_file():
                continue
            rel_path = file_path.relative_to(path).as_posix()
            digest.update(rel_path.encode("utf-8"))
            self._update_digest_with_file(digest, file_path)

        return digest.hexdigest()

    @staticmethod
    def _update_digest_with_file(digest: Any, file_path: Path) -> None:
        """Incrementally hash file content."""
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                digest.update(chunk)
