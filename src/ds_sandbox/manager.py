"""
ds-sandbox core manager

Sandbox manager is the main orchestrator for code execution.
It manages backends, workspaces, and execution lifecycle.
"""

import asyncio
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ds_sandbox.backends import SandboxBackend
from ds_sandbox.config import SandboxConfig
from ds_sandbox.security.scanner import CodeScanner
from ds_sandbox.path_utils import validate_path_component
from ds_sandbox.errors import (
    BackendUnavailableError,
    DatasetNotFoundError,
    ExecutionFailedError,
    ExecutionNotFoundError,
    ExecutionTimeoutError,
    InvalidRequestError,
    SandboxError,
    WorkspaceNotFoundError,
)
from ds_sandbox.types import (
    CodeScanResult,
    ExecutionRequest,
    ExecutionResult,
    Workspace,
    SandboxEvent,
    PausedWorkspace,
    SandboxMetrics,
    Template,
    StorageConfig,
)
from ds_sandbox.execution.tracker import ExecutionTracker
from ds_sandbox.execution.router import IsolationRouter
from ds_sandbox.template_manager.manager import TemplateManager

logger = logging.getLogger(__name__)


def _copy_file(src: Path, dst: Path) -> None:
    """Copy a single file, creating parent directories if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


class SandboxManager:
    """
    Main sandbox manager - single entry point for all operations.

    Responsibilities:
    - Backend registration and routing
    - Workspace lifecycle management
    - Execution orchestration
    - Security policy enforcement
    """

    # Default backend registry mapping backend names to classes
    DEFAULT_BACKENDS: Dict[str, Type[SandboxBackend]] = {}

    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize sandbox manager.

        Args:
            config: Optional sandbox configuration. If not provided,
                   uses default configuration.
        """
        if config is None:
            config = SandboxConfig()

        self.config = config
        self._backends: Dict[str, SandboxBackend] = {}
        self._router = IsolationRouter(config)
        self._tracker = ExecutionTracker()
        self._template_manager = TemplateManager(
            Path(config.workspace_base_dir).parent / "templates"
        )
        self._workspace_cache: Dict[str, Workspace] = {}
        self._paused_workspaces: Dict[str, PausedWorkspace] = {}
        self._storage_mounts: Dict[str, Dict[str, StorageConfig]] = {}  # workspace_id -> {mount_name -> StorageConfig}

        # Ensure paused workspaces directory exists
        self._paused_dir = Path(config.paused_workspaces_base_dir)
        self._paused_dir.mkdir(parents=True, exist_ok=True)

        # Load default backends
        self.load_backends_from_registry()

        # Load existing templates from disk
        self._template_manager._load_templates()

        logger.info(f"SandboxManager initialized with config: default_backend={config.default_backend}")

    def register_backend(self, name: str, backend: SandboxBackend) -> None:
        """
        Register a sandbox backend.

        Args:
            name: Backend name identifier
            backend: SandboxBackend instance

        Raises:
            TypeError: If backend is not a SandboxBackend instance
        """
        if not isinstance(backend, SandboxBackend):
            raise TypeError(f"Backend must be a SandboxBackend instance, got {type(backend)}")

        self._backends[name] = backend
        logger.info(f"Registered backend: {name}")

    def register_backend_class(
        self,
        name: str,
        backend_class: Type[SandboxBackend],
        **init_kwargs
    ) -> None:
        """
        Register a backend class by instantiating it.

        Args:
            name: Backend name identifier
            backend_class: SandboxBackend class to instantiate
            **init_kwargs: Keyword arguments passed to backend constructor
        """
        backend = backend_class(**init_kwargs)
        self.register_backend(name, backend)

    def get_backend(self, name: str) -> SandboxBackend:
        """
        Get a registered backend by name.

        Args:
            name: Backend name

        Returns:
            SandboxBackend instance

        Raises:
            BackendUnavailableError: If backend is not registered
        """
        if name not in self._backends:
            raise BackendUnavailableError(backend=name)

        return self._backends[name]

    def load_backends_from_registry(self) -> None:
        """
        Load all default backends from the registry.

        This method dynamically imports and registers backend classes
        based on the DEFAULT_BACKENDS mapping.
        """
        from ds_sandbox.backends.docker import DockerSandbox
        from ds_sandbox.backends.local import LocalSubprocessSandbox

        # Register built-in Docker backend
        self.register_backend_class("docker", DockerSandbox)
        # Register built-in local subprocess backend (manual selection only)
        self.register_backend_class("local", LocalSubprocessSandbox)

        logger.info(f"Loaded {len(self._backends)} backends from registry")

    async def execute(
        self,
        request: ExecutionRequest,
    ) -> ExecutionResult:
        """
        Execute code in the specified workspace.

        This is the main entry point for code execution.

        Flow:
        1. Validate workspace exists
        2. Prepare datasets in workspace/data/
        3. Select backend based on mode/risk assessment
        4. Mount workspace to sandbox
        5. Execute code in isolated environment
        6. Collect results and artifacts
        7. Write audit log

        Args:
            request: Execution request with all parameters

        Returns:
            ExecutionResult with execution details

        Raises:
            WorkspaceNotFoundError: If workspace doesn't exist
            BackendUnavailableError: If selected backend is not available
            DatasetNotFoundError: If requested dataset doesn't exist
            ExecutionTimeoutError: If execution times out
            ExecutionFailedError: If execution fails
        """
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        start_time = datetime.now(timezone.utc)

        logger.info(f"Starting execution {execution_id} in workspace {request.workspace_id}")

        try:
            # Step 1: Validate and get workspace
            workspace = await self.get_workspace(request.workspace_id)

            # Step 2: Validate request
            self._router._validate_request(request)

            # Step 3: Prepare datasets if requested
            if request.datasets:
                await self._prepare_datasets(request.datasets, workspace)

            # Step 4: Perform security scan if needed
            code_scan_result = None
            if self._should_scan_code(request):
                code_scan_result = await self._scan_code(request.code)
                logger.info(
                    f"Code scan result: risk_score={code_scan_result.risk_score}, "
                    f"recommended_backend={code_scan_result.recommended_backend}"
                )

            # Step 5: Select backend based on request and scan result
            backend_name = self._router.decide_backend(request, code_scan_result)
            backend = self.get_backend(backend_name)

            logger.info(f"Selected backend '{backend_name}' for execution {execution_id}")

            # Track execution start
            self._tracker._track_execution_start(execution_id, request.workspace_id, backend_name)

            # Step 6: Execute code
            result = await backend.execute(request, workspace)

            # Step 7: Update result with execution metadata
            result.execution_id = execution_id
            result.workspace_id = request.workspace_id
            result.backend = backend_name
            result.isolation_level = self._get_actual_isolation_level(
                request, code_scan_result
            )

            # Calculate duration
            end_time = datetime.now(timezone.utc)
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
            result.duration_ms = duration_ms

            # Step 8: Collect artifacts from workspace
            result.artifacts = await self._collect_artifacts(workspace)

            logger.info(
                f"Execution {execution_id} completed: success={result.success}, "
                f"duration_ms={duration_ms}"
            )

            # Track execution completion
            status = "completed" if result.success else "failed"
            self._tracker._track_execution_complete(
                execution_id,
                status,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                duration_ms=duration_ms,
            )

            return result

        except SandboxError:
            # Re-raise sandbox errors as-is
            raise
        except asyncio.TimeoutError:
            error = ExecutionTimeoutError(
                execution_id=execution_id,
                timeout_sec=request.timeout_sec
            )
            logger.error(f"Execution {execution_id} timed out: {error}")
            self._tracker._track_execution_complete(execution_id, "failed")
            raise error
        except Exception as e:
            logger.error(f"Execution {execution_id} failed: {e}", exc_info=True)
            self._tracker._track_execution_complete(execution_id, "failed")
            raise ExecutionFailedError(
                execution_id=execution_id,
                reason=str(e)
            )

    async def get_execution_status(
        self,
        workspace_id: str,
        execution_id: str,
    ) -> Dict[str, Any]:
        """
        Get the status of an execution.

        Args:
            workspace_id: Workspace identifier
            execution_id: Execution identifier

        Returns:
            Dictionary with execution status

        Raises:
            SandboxError: If execution not found
        """
        status = self._tracker.get_execution_status(execution_id)
        # Verify workspace matches
        if status["workspace_id"] != workspace_id:
            raise ExecutionNotFoundError(execution_id=execution_id)

        return status

    async def stop_execution(
        self,
        workspace_id: str,
        execution_id: str,
    ) -> bool:
        """
        Stop a running execution.

        Args:
            workspace_id: Workspace identifier
            execution_id: Execution identifier

        Returns:
            True if execution was stopped, False if it was already completed

        Raises:
            ExecutionNotFoundError: If execution not found
        """
        status = self._tracker.get_execution_status(execution_id)
        # Verify workspace matches
        if status["workspace_id"] != workspace_id:
            raise ExecutionNotFoundError(execution_id=execution_id)

        # Check if execution is still running
        if status["status"] != "running":
            logger.info(f"Execution {execution_id} is not running (status: {status['status']})")
            return False

        backend_name = status.get("backend")
        if backend_name:
            backend = self.get_backend(backend_name)
            # Stop the backend execution if supported
            if hasattr(backend, "stop_execution"):
                await backend.stop_execution(execution_id, workspace_id)

        # Update execution status
        self._tracker._track_execution_complete(execution_id, "stopped")
        logger.info(f"Execution {execution_id} stopped")

        return True

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics.

        Returns:
            Dictionary with system metrics
        """
        return self._tracker.get_metrics()

    def get_workspace_metrics(self, workspace_id: str) -> List[SandboxMetrics]:
        """
        Get metrics history for a workspace.

        Args:
            workspace_id: Workspace identifier

        Returns:
            List of SandboxMetrics objects
        """
        return self._tracker.get_workspace_metrics(workspace_id)

    def get_system_metrics(self) -> List[SandboxMetrics]:
        """
        Get all system metrics across all workspaces.

        Returns:
            List of all SandboxMetrics objects
        """
        return self._tracker.get_system_metrics()

    def collect_workspace_metrics(self, workspace_id: str) -> SandboxMetrics:
        """
        Collect current metrics for a workspace.

        Args:
            workspace_id: Workspace identifier

        Returns:
            SandboxMetrics object with current metrics
        """
        return self._tracker.collect_workspace_metrics(workspace_id)

    # =========================================================================
    # Event Management
    # =========================================================================

    def emit_event(
        self,
        event_type: str,
        workspace_id: str,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> SandboxEvent:
        """
        Emit a sandbox lifecycle event.

        Args:
            event_type: Type of event (e.g., "sandbox.lifecycle.created")
            workspace_id: Workspace ID
            event_data: Additional event data

        Returns:
            The created SandboxEvent
        """
        return self._tracker.emit_event(event_type, workspace_id, event_data)

    def get_events(
        self,
        workspace_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[SandboxEvent]:
        """
        Get sandbox lifecycle events.

        Args:
            workspace_id: Optional workspace ID to filter events
            limit: Maximum number of events to return (default: 10)

        Returns:
            List of SandboxEvent objects
        """
        return self._tracker.get_events(workspace_id, limit)

    def get_all_events(self, limit: int = 100) -> List[SandboxEvent]:
        """
        Get all sandbox lifecycle events (admin endpoint).

        Args:
            limit: Maximum number of events to return (default: 100)

        Returns:
            List of all SandboxEvent objects
        """
        return self._tracker.get_all_events(limit)

    async def create_workspace(
        self,
        workspace_id: str,
        setup_dirs: Optional[List[str]] = None,
        external_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Workspace:
        """
        Create a new workspace.

        Creates directory structure:
        {workspace_base_dir}/{workspace_id}/
            ├── data/
            ├── models/
            ├── outputs/
            └── .workspace/

        Or creates a symlink to an external path for local backend:
        {workspace_base_dir}/{workspace_id} -> external_path

        Args:
            workspace_id: Unique workspace identifier
            setup_dirs: Additional subdirectories to create
            external_path: If provided, create symlink to this path instead of real directory (for local backend)
            metadata: Custom metadata for the sandbox

        Returns:
            Workspace object with workspace information

        Raises:
            FileExistsError: If workspace already exists
            PermissionError: If unable to create directories
        """
        from typing import Dict, Any

        # Validate workspace_id against path traversal
        validate_path_component(workspace_id, "workspace_id")

        base_dir = Path(self.config.workspace_base_dir)
        workspace_path = base_dir / workspace_id

        logger.info(f"Creating workspace: {workspace_id} at {workspace_path}")

        # Check if workspace already exists
        if workspace_path.exists():
            raise FileExistsError(f"Workspace '{workspace_id}' already exists")

        # Validate setup_dirs against path traversal
        for d in (setup_dirs or []):
            validate_path_component(d, "setup_dirs")

        try:
            # If external_path is provided, create symlink (for local backend)
            if external_path:
                external = Path(external_path)
                # Create parent directory if needed
                workspace_path.parent.mkdir(parents=True, exist_ok=True)
                # Create symlink to external path
                workspace_path.symlink_to(external.resolve())
                logger.info(f"Created symlink: {workspace_path} -> {external}")
            else:
                # Create workspace directory (just the base directory)
                workspace_path.mkdir(parents=True, exist_ok=True)

                # Only create user-specified subdirectories
                if setup_dirs:
                    for subdir in setup_dirs:
                        subdir_path = workspace_path / subdir
                        subdir_path.mkdir(parents=True, exist_ok=True)

            # Create .workspace metadata file (even for symlink)
            metadata_path = workspace_path / ".workspace"
            metadata_path.touch()

            # Create workspace object (no default subdirs)
            now = datetime.now(timezone.utc).isoformat()
            workspace = Workspace(
                workspace_id=workspace_id,
                host_path=str(workspace_path),
                guest_path="/workspace",
                subdirs=setup_dirs or [],
                status="ready",
                created_at=now,
                last_used_at=None,
                metadata=metadata or {},
            )

            # Cache workspace
            self._workspace_cache[workspace_id] = workspace

            # Emit lifecycle event
            self.emit_event(
                event_type="sandbox.lifecycle.created",
                workspace_id=workspace_id,
                event_data={"host_path": str(workspace_path), "guest_path": "/workspace"},
            )

            logger.info(f"Workspace '{workspace_id}' created successfully")

            return workspace

        except PermissionError as e:
            logger.error(f"Permission denied creating workspace {workspace_id}: {e}")
            raise
        except Exception as e:
            # Cleanup on failure
            if workspace_path.exists():
                shutil.rmtree(workspace_path, ignore_errors=True)
            logger.error(f"Failed to create workspace {workspace_id}: {e}", exc_info=True)
            raise

    async def get_workspace(self, workspace_id: str) -> Workspace:
        """
        Get workspace information.

        Args:
            workspace_id: Workspace ID

        Returns:
            Workspace object

        Raises:
            WorkspaceNotFoundError: If workspace doesn't exist
        """
        # Check cache first
        if workspace_id in self._workspace_cache:
            return self._workspace_cache[workspace_id]

        base_dir = Path(self.config.workspace_base_dir)
        workspace_path = base_dir / workspace_id

        if not workspace_path.exists():
            raise WorkspaceNotFoundError(workspace_id=workspace_id)

        # Load workspace from disk
        metadata_path = workspace_path / ".workspace"
        if metadata_path.exists():
            # Workspace exists and has metadata
            subdirs = []
            for item in workspace_path.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    subdirs.append(item.name)

            now = datetime.now(timezone.utc).isoformat()
            workspace = Workspace(
                workspace_id=workspace_id,
                host_path=str(workspace_path),
                guest_path="/workspace",
                subdirs=subdirs or ["data", "models", "outputs"],
                status="ready",
                created_at=now,
                last_used_at=now,
            )
        else:
            # Legacy workspace without metadata
            raise WorkspaceNotFoundError(workspace_id=workspace_id)

        # Cache workspace
        self._workspace_cache[workspace_id] = workspace

        return workspace

    async def delete_workspace(self, workspace_id: str, force: bool = False) -> None:
        """
        Delete a workspace and all its data.

        Args:
            workspace_id: Workspace ID to delete
            force: If True, ignore errors during deletion

        Raises:
            WorkspaceNotFoundError: If workspace doesn't exist
        """
        workspace = await self.get_workspace(workspace_id)
        workspace_path = Path(workspace.host_path)

        logger.info(f"Deleting workspace: {workspace_id}")

        if not workspace_path.exists():
            raise WorkspaceNotFoundError(workspace_id=workspace_id)

        try:
            # Remove from cache
            self._workspace_cache.pop(workspace_id, None)

            # Delete directory tree
            if force:
                shutil.rmtree(workspace_path, ignore_errors=True)
            else:
                shutil.rmtree(workspace_path)

            # Emit lifecycle event
            self.emit_event(
                event_type="sandbox.lifecycle.killed",
                workspace_id=workspace_id,
                event_data={"host_path": str(workspace_path)},
            )

            logger.info(f"Workspace '{workspace_id}' deleted successfully")

        except Exception as e:
            logger.error(f"Failed to delete workspace {workspace_id}: {e}", exc_info=True)
            if not force:
                raise

    async def health_check(self) -> Dict[str, dict]:
        """
        Check health status of all backends.

        Returns:
            Dict mapping backend names to their health status
        """
        health_status = {}

        for backend_name, backend in self._backends.items():
            try:
                status = await backend.health_check()
                health_status[backend_name] = {
                    "status": status.get("status", "unknown"),
                    "backend": backend_name,
                    "details": status,
                }
            except Exception as e:
                health_status[backend_name] = {
                    "status": "error",
                    "backend": backend_name,
                    "error": str(e),
                }

        return health_status

    async def list_workspaces(
        self,
        state: Optional[str] = None,
    ) -> List[Workspace]:
        """
        List all existing workspaces.

        Args:
            state: Optional filter by state ("running" or "paused")

        Returns:
            List of Workspace objects
        """
        base_dir = Path(self.config.workspace_base_dir)

        if not base_dir.exists():
            return []

        workspaces = []
        for entry in base_dir.iterdir():
            if entry.is_dir():
                try:
                    workspace = await self.get_workspace(entry.name)
                    # Filter by state if specified
                    if state == "paused":
                        # Check if workspace is in paused list
                        if workspace.workspace_id in self._paused_workspaces:
                            workspaces.append(workspace)
                    elif state == "running":
                        # Only include if not paused
                        if workspace.workspace_id not in self._paused_workspaces:
                            workspaces.append(workspace)
                    else:
                        # No filter, include all
                        workspaces.append(workspace)
                except WorkspaceNotFoundError:
                    continue

        return workspaces

    async def cleanup_expired_workspaces(self) -> int:
        """
        Remove workspaces that have exceeded retention period.

        Returns:
            Number of workspaces cleaned up
        """
        from datetime import timedelta

        workspaces = await self.list_workspaces()
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config.workspace_retention_days)

        cleaned = 0
        for workspace in workspaces:
            if workspace.last_used_at:
                last_used = datetime.fromisoformat(workspace.last_used_at.replace("Z", "+00:00"))
                if last_used < cutoff_date:
                    await self.delete_workspace(workspace.workspace_id, force=True)
                    cleaned += 1

        logger.info(f"Cleaned up {cleaned} expired workspaces")
        return cleaned

    # =========================================================================
    # Workspace Pause/Resume (E2B-compatible)
    # =========================================================================

    async def pause_workspace(self, workspace_id: str) -> PausedWorkspace:
        """
        Pause a workspace and save its state.

        Saves the workspace filesystem to a backup directory. The paused workspace
        can be resumed later to restore its state.

        Args:
            workspace_id: Workspace ID to pause

        Returns:
            PausedWorkspace object with backup information

        Raises:
            WorkspaceNotFoundError: If workspace doesn't exist
        """
        from datetime import timedelta

        workspace = await self.get_workspace(workspace_id)
        workspace_path = Path(workspace.host_path)

        if not workspace_path.exists():
            raise WorkspaceNotFoundError(workspace_id=workspace_id)

        # Create backup directory with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = self._paused_dir / f"{workspace_id}-{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Pausing workspace {workspace_id}, backing up to {backup_dir}")

        # Copy workspace files to backup directory
        import shutil as _shutil
        file_count = 0
        total_size = 0

        if workspace_path.exists():
            for item in workspace_path.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(workspace_path)
                    dest_path = backup_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    _shutil.copy2(item, dest_path)
                    file_count += 1
                    total_size += item.stat().st_size

        # Calculate expiry date (30 days by default)
        paused_at = datetime.now(timezone.utc)
        expires_at = paused_at + timedelta(days=self.config.paused_workspace_retention_days)

        # Create paused workspace metadata
        paused_workspace = PausedWorkspace(
            workspace_id=workspace_id,
            backup_path=str(backup_dir),
            original_path=str(workspace_path),
            paused_at=paused_at.isoformat(),
            expires_at=expires_at.isoformat(),
            files_count=file_count,
            size_mb=total_size / (1024 * 1024),
            metadata={
                "original_host_path": workspace.host_path,
                "original_guest_path": workspace.guest_path,
                "original_subdirs": workspace.subdirs,
            }
        )

        # Store in memory
        self._paused_workspaces[workspace_id] = paused_workspace

        # Emit lifecycle event
        self.emit_event(
            event_type="sandbox.lifecycle.paused",
            workspace_id=workspace_id,
            event_data={
                "backup_path": str(backup_dir),
                "files_count": file_count,
                "size_mb": total_size / (1024 * 1024),
            },
        )

        logger.info(f"Workspace {workspace_id} paused successfully: {file_count} files, {total_size / (1024 * 1024):.2f} MB")
        return paused_workspace

    async def resume_workspace(self, workspace_id: str) -> Workspace:
        """
        Resume a paused workspace from saved state.

        Restores the workspace from its backup directory to the original location.

        Args:
            workspace_id: Workspace ID to resume (must be paused)

        Returns:
            Workspace object with restored information

        Raises:
            WorkspaceNotFoundError: If paused workspace doesn't exist
        """
        if workspace_id not in self._paused_workspaces:
            raise WorkspaceNotFoundError(workspace_id=workspace_id)

        paused_workspace = self._paused_workspaces[workspace_id]
        backup_path = Path(paused_workspace.backup_path)
        original_path = Path(paused_workspace.original_path)

        if not backup_path.exists():
            raise WorkspaceNotFoundError(workspace_id=workspace_id)

        logger.info(f"Resuming workspace {workspace_id} from {backup_path}")

        # Restore files from backup
        import shutil as _shutil

        # Remove original workspace if it exists
        if original_path.exists() or original_path.is_symlink():
            if original_path.is_symlink():
                original_path.unlink()
            else:
                _shutil.rmtree(original_path)

        # Recreate original directory
        original_path.mkdir(parents=True, exist_ok=True)

        # Copy files from backup
        for item in backup_path.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(backup_path)
                dest_path = original_path / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                _shutil.copy2(item, dest_path)

        # Ensure .workspace metadata file exists
        metadata_path = original_path / ".workspace"
        if not metadata_path.exists():
            metadata_path.touch()

        # Remove from paused workspaces
        del self._paused_workspaces[workspace_id]

        # Emit lifecycle event
        self.emit_event(
            event_type="sandbox.lifecycle.resumed",
            workspace_id=workspace_id,
            event_data={"original_path": str(original_path)},
        )

        # Get or create workspace object
        workspace = await self.get_workspace(workspace_id)
        logger.info(f"Workspace {workspace_id} resumed successfully")

        return workspace

    async def get_paused_workspace(self, workspace_id: str) -> PausedWorkspace:
        """
        Get paused workspace metadata.

        Args:
            workspace_id: Workspace ID

        Returns:
            PausedWorkspace object

        Raises:
            WorkspaceNotFoundError: If workspace is not paused
        """
        if workspace_id not in self._paused_workspaces:
            raise WorkspaceNotFoundError(workspace_id=workspace_id)

        return self._paused_workspaces[workspace_id]

    async def list_paused_workspaces(self) -> List[PausedWorkspace]:
        """
        List all paused workspaces.

        Returns:
            List of PausedWorkspace objects
        """
        return list(self._paused_workspaces.values())

    async def delete_paused_workspace(self, workspace_id: str) -> None:
        """
        Delete a paused workspace backup.

        Args:
            workspace_id: Workspace ID

        Raises:
            WorkspaceNotFoundError: If workspace is not paused
        """
        if workspace_id not in self._paused_workspaces:
            raise WorkspaceNotFoundError(workspace_id=workspace_id)

        paused_workspace = self._paused_workspaces[workspace_id]
        backup_path = Path(paused_workspace.backup_path)

        # Remove backup directory
        if backup_path.exists():
            import shutil as _shutil
            _shutil.rmtree(backup_path)

        # Remove from memory
        del self._paused_workspaces[workspace_id]

        logger.info(f"Paused workspace {workspace_id} deleted")

    async def cleanup_expired_paused_workspaces(self) -> int:
        """
        Remove paused workspaces that have exceeded retention period.

        Returns:
            Number of paused workspaces cleaned up
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        cleaned = 0

        for workspace_id, paused_workspace in list(self._paused_workspaces.items()):
            expires_at = datetime.fromisoformat(paused_workspace.expires_at.replace("Z", "+00:00"))
            if expires_at < now:
                await self.delete_paused_workspace(workspace_id)
                cleaned += 1

        logger.info(f"Cleaned up {cleaned} expired paused workspaces")
        return cleaned

    # =========================================================================
    # Storage Management
    # =========================================================================

    async def mount_storage(
        self,
        workspace_id: str,
        storage_config: StorageConfig,
        mount_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Mount storage to a workspace.

        This method mounts cloud storage (S3, GCS, Azure) to a workspace,
        making the storage bucket accessible from within the sandbox.

        Args:
            workspace_id: Workspace identifier
            storage_config: Storage configuration (provider, bucket, credentials, etc.)
            mount_name: Optional name for this mount (defaults to bucket name)

        Returns:
            Dictionary with mount information including mount_point

        Raises:
            WorkspaceNotFoundError: If workspace doesn't exist
        """
        # Validate workspace exists
        workspace = await self.get_workspace(workspace_id)

        # Generate mount name if not provided
        if mount_name is None:
            mount_name = storage_config.bucket

        # Validate mount_name against path traversal
        validate_path_component(mount_name, "mount_name")

        # Initialize storage mounts dict for this workspace if needed
        if workspace_id not in self._storage_mounts:
            self._storage_mounts[workspace_id] = {}

        # Check if mount already exists
        if mount_name in self._storage_mounts[workspace_id]:
            logger.warning(f"Storage mount '{mount_name}' already exists in workspace {workspace_id}")

        # Determine mount point
        mount_point = storage_config.mount_point
        if mount_point is None:
            mount_point = f"{self.config.storage_base_dir}/{mount_name}"

        # Store the storage config
        self._storage_mounts[workspace_id][mount_name] = storage_config

        # Emit lifecycle event
        self.emit_event(
            event_type="storage.mounted",
            workspace_id=workspace_id,
            event_data={
                "mount_name": mount_name,
                "provider": storage_config.provider,
                "bucket": storage_config.bucket,
                "mount_point": mount_point,
                "read_only": storage_config.read_only,
            },
        )

        logger.info(
            f"Mounted storage '{mount_name}' (bucket: {storage_config.bucket}) "
            f"to workspace {workspace_id} at {mount_point}"
        )

        return {
            "mount_name": mount_name,
            "provider": storage_config.provider,
            "bucket": storage_config.bucket,
            "mount_point": mount_point,
            "path_prefix": storage_config.path_prefix,
            "read_only": storage_config.read_only,
        }

    async def unmount_storage(
        self,
        workspace_id: str,
        mount_name: str,
    ) -> Dict[str, str]:
        """
        Unmount storage from a workspace.

        Args:
            workspace_id: Workspace identifier
            mount_name: Name of the mount to remove

        Returns:
            Dictionary with status information

        Raises:
            WorkspaceNotFoundError: If workspace doesn't exist
        """
        # Validate workspace exists
        await self.get_workspace(workspace_id)

        # Check if storage mounts exist for this workspace
        if workspace_id not in self._storage_mounts:
            raise SandboxError(
                message=f"No storage mounts found in workspace {workspace_id}",
                error_code="STORAGE_NOT_FOUND",
            )

        # Check if mount exists
        if mount_name not in self._storage_mounts[workspace_id]:
            raise SandboxError(
                message=f"Storage mount '{mount_name}' not found in workspace {workspace_id}",
                error_code="STORAGE_NOT_FOUND",
            )

        # Get the storage config for the event
        storage_config = self._storage_mounts[workspace_id][mount_name]

        # Remove the mount
        del self._storage_mounts[workspace_id][mount_name]

        # Clean up empty dict
        if not self._storage_mounts[workspace_id]:
            del self._storage_mounts[workspace_id]

        # Emit lifecycle event
        self.emit_event(
            event_type="storage.unmounted",
            workspace_id=workspace_id,
            event_data={
                "mount_name": mount_name,
                "provider": storage_config.provider,
                "bucket": storage_config.bucket,
            },
        )

        logger.info(f"Unmounted storage '{mount_name}' from workspace {workspace_id}")

        return {
            "status": "unmounted",
            "mount_name": mount_name,
        }

    async def list_storage_mounts(
        self,
        workspace_id: str,
    ) -> List[Dict[str, Any]]:
        """
        List storage mounts for a workspace.

        Args:
            workspace_id: Workspace identifier

        Returns:
            List of storage mount configurations

        Raises:
            WorkspaceNotFoundError: If workspace doesn't exist
        """
        # Validate workspace exists
        await self.get_workspace(workspace_id)

        # Return empty list if no mounts
        if workspace_id not in self._storage_mounts:
            return []

        mounts = []
        for mount_name, config in self._storage_mounts[workspace_id].items():
            mount_point = config.mount_point
            if mount_point is None:
                mount_point = f"{self.config.storage_base_dir}/{mount_name}"

            mounts.append({
                "mount_name": mount_name,
                "provider": config.provider,
                "bucket": config.bucket,
                "mount_point": mount_point,
                "path_prefix": config.path_prefix,
                "read_only": config.read_only,
            })

        return mounts

    async def get_storage_mount(
        self,
        workspace_id: str,
        mount_name: str,
    ) -> Dict[str, Any]:
        """
        Get storage mount configuration for a workspace.

        Args:
            workspace_id: Workspace identifier
            mount_name: Name of the mount

        Returns:
            Storage mount configuration (without credentials)

        Raises:
            WorkspaceNotFoundError: If workspace doesn't exist
            SandboxError: If mount not found
        """
        # Validate workspace exists
        await self.get_workspace(workspace_id)

        # Check if storage mounts exist
        if workspace_id not in self._storage_mounts:
            raise SandboxError(
                message=f"No storage mounts found in workspace {workspace_id}",
                error_code="STORAGE_NOT_FOUND",
            )

        # Check if mount exists
        if mount_name not in self._storage_mounts[workspace_id]:
            raise SandboxError(
                message=f"Storage mount '{mount_name}' not found in workspace {workspace_id}",
                error_code="STORAGE_NOT_FOUND",
            )

        config = self._storage_mounts[workspace_id][mount_name]
        mount_point = config.mount_point
        if mount_point is None:
            mount_point = f"{self.config.storage_base_dir}/{mount_name}"

        return {
            "mount_name": mount_name,
            "provider": config.provider,
            "bucket": config.bucket,
            "mount_point": mount_point,
            "path_prefix": config.path_prefix,
            "read_only": config.read_only,
        }

    async def _prepare_datasets(
        self,
        dataset_names: List[str],
        workspace: Workspace,
        strategy: str = "copy",
    ) -> None:
        """
        Prepare datasets in workspace/data/ directory.

        Args:
            dataset_names: List of dataset names to prepare
            workspace: Target workspace
            strategy: Preparation strategy - "copy" or "link"

        Raises:
            DatasetNotFoundError: If dataset doesn't exist in registry
        """
        dataset_registry = Path(self.config.dataset_registry_dir)
        workspace_data_dir = Path(workspace.host_path) / "data"

        for dataset_name in dataset_names:
            # Validate dataset_name against path traversal
            validate_path_component(dataset_name, "dataset_name")

            dataset_path = dataset_registry / dataset_name

            if not dataset_path.exists():
                raise DatasetNotFoundError(dataset_name=dataset_name)

            # Create symlink or copy based on strategy
            dest_path = workspace_data_dir / dataset_name
            is_file = dataset_path.is_file()

            if strategy == "link" and not is_file:
                if dest_path.exists() or dest_path.is_symlink():
                    dest_path.unlink()

                try:
                    dest_path.symlink_to(dataset_path)
                    logger.debug(f"Created symlink for dataset: {dataset_name}")
                except OSError:
                    # Fall back to copy if symlink fails (e.g., cross-device)
                    shutil.copytree(dataset_path, dest_path)
                    logger.debug(f"Copied dataset (symlink failed): {dataset_name}")
            else:
                # Copy strategy or file-type dataset
                if dest_path.exists():
                    if dest_path.is_dir() and not dest_path.is_symlink():
                        shutil.rmtree(dest_path)
                    else:
                        dest_path.unlink()

                if is_file:
                    _copy_file(dataset_path, dest_path)
                else:
                    shutil.copytree(dataset_path, dest_path)
                logger.debug(f"Copied dataset: {dataset_name}")

    async def _collect_artifacts(self, workspace: Workspace) -> List[str]:
        """
        Collect artifacts from workspace/outputs/ directory.

        Args:
            workspace: Workspace to collect artifacts from

        Returns:
            List of artifact file paths (relative to workspace)
        """
        artifacts = []
        outputs_dir = Path(workspace.host_path) / "outputs"

        if not outputs_dir.exists():
            return artifacts

        for artifact_path in outputs_dir.rglob("*"):
            if artifact_path.is_file():
                rel_path = artifact_path.relative_to(workspace.host_path)
                artifacts.append(str(rel_path))

        return artifacts

    async def _scan_code(self, code: str) -> CodeScanResult:
        """
        Scan code for security issues.

        Uses CodeScanner to perform static analysis on Python code
        and determine risk score and recommended backend.

        Args:
            code: Python code to scan

        Returns:
            CodeScanResult with scan findings
        """
        scanner = CodeScanner()
        return scanner.scan(code)

    def _should_scan_code(self, request: ExecutionRequest) -> bool:
        """
        Determine if code should be scanned before execution.

        Args:
            request: Execution request

        Returns:
            True if code should be scanned
        """
        # Scan code when network access is requested
        return request.network_policy in ("whitelist", "proxy")

    def _get_actual_isolation_level(
        self,
        request: ExecutionRequest,
        scan_result: Optional[CodeScanResult],
    ) -> str:
        """
        Determine the actual isolation level used.

        Args:
            request: Original execution request
            scan_result: Code scan result if available

        Returns:
            Isolation level string
        """
        if scan_result and scan_result.recommended_backend:
            return scan_result.recommended_backend

        return request.mode

    # =========================================================================
    # Template Management
    # =========================================================================

    async def build_template(
        self,
        template: Template,
        alias: Optional[str] = None,
        wait_timeout: int = 60,
        debug: bool = False,
    ) -> Template:
        """
        Build a template.

        This method stores the template configuration and optionally builds
        the actual Docker image if the backend supports it.

        Args:
            template: Template configuration
            alias: Primary alias for the template
            wait_timeout: Wait timeout in seconds
            debug: Enable debug mode

        Returns:
            Template object with assigned ID

        Example:
            >>> from ds_sandbox.template import TemplateBuilder
            >>> template = (TemplateBuilder()
            ...     .from_python_image("3.11")
            ...     .set_envs({"MY_VAR": "value"})
            ...     .build("my-template"))
            >>> manager = SandboxManager()
            >>> result = await manager.build_template(template)
        """
        result = await self._template_manager.build_template(
            template, alias, wait_timeout, debug
        )

        # Emit lifecycle event
        self.emit_event(
            event_type="template.built",
            workspace_id="",
            event_data={
                "template_id": template.id,
                "alias": alias,
                "image": template.image,
                "debug": debug,
            },
        )

        logger.info(f"Template built: {template.id}")
        return result

    async def list_templates(self) -> List[Template]:
        """
        List all available templates.

        Returns:
            List of Template objects

        Example:
            >>> templates = await manager.list_templates()
            >>> for t in templates:
            ...     print(f"{t.id}: {t.name}")
        """
        return await self._template_manager.list_templates()

    async def get_template(self, template_id: str) -> Template:
        """
        Get template by ID or alias.

        Args:
            template_id: Template ID or alias

        Returns:
            Template object

        Raises:
            SandboxError: If template not found
        """
        return await self._template_manager.get_template(template_id)

    async def delete_template(self, template_id: str) -> None:
        """
        Delete a template.

        Args:
            template_id: Template ID to delete

        Raises:
            SandboxError: If template not found
        """
        await self._template_manager.delete_template(template_id)

        # Emit lifecycle event
        self.emit_event(
            event_type="template.deleted",
            workspace_id="",
            event_data={"template_id": template_id},
        )

        logger.info(f"Template deleted: {template_id}")
