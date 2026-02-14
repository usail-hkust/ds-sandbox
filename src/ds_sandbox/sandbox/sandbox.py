"""
E2B-compatible Sandbox API for ds-sandbox.

This module provides a simple, E2B-compatible interface for code execution.
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ds_sandbox.config import SandboxConfig
from ds_sandbox.manager import SandboxManager
from ds_sandbox.types import ExecutionRequest, SandboxInfo, Template, PausedWorkspace, SandboxMetrics
from ds_sandbox.api.sdk import SandboxSDK

# Import internal classes
from ds_sandbox.sandbox.files import Files
from ds_sandbox.sandbox.commands import Commands
from ds_sandbox.sandbox.result import CodeResult, ExecutionLogs

# Warning constants
JAVASCRIPT_WARNING = (
    "ds-sandbox only supports Python execution. "
    "JavaScript/Node.js execution is not supported. "
    "Use Python for code execution in ds-sandbox."
)

logger = logging.getLogger(__name__)


class RemoteSandbox:
    """
    Remote Sandbox that connects to ds-sandbox API server.

    Usage:
        >>> from ds_sandbox import Sandbox
        >>>
        >>> # Connect to remote API server
        >>> sandbox = Sandbox.create(
        ...     config=SandboxConfig(api_endpoint="http://192.168.1.100:8000")
        ... )
        >>>
        >>> result = sandbox.run_code("print('hello')")
        >>> print(result.stdout)
        >>>
        >>> sandbox.kill()
    """

    def __init__(
        self,
        workspace_id: str,
        sdk: SandboxSDK,
        workspace: Any,
        config: SandboxConfig,
    ):
        self.workspace_id = workspace_id
        self._sdk = sdk
        self.workspace = workspace
        self.config = config
        self._files = Files(self)
        self.sandbox_id = workspace_id

    @property
    def files(self) -> Files:
        return self._files

    def run_code(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> CodeResult:
        """Run Python code synchronously via remote API."""
        return asyncio.run(self.run_code_async(code, timeout=timeout))

    async def run_code_async(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> CodeResult:
        """Run Python code asynchronously via remote API."""
        # Use execute() with wait=True (default) to get full result
        result = await self._sdk.execute(
            workspace_id=self.workspace_id,
            code=code,
            timeout_sec=timeout or 3600,
        )
        return CodeResult(
            logs=ExecutionLogs(stdout=result.stdout, stderr=result.stderr),
            error=result.stderr if not result.success else None,
            results=None,
            _stdout=result.stdout,
            _stderr=result.stderr,
        )

    async def kill(self):
        """Kill the sandbox (stop workspace)."""
        await self._sdk.delete_workspace(self.workspace_id)

    def kill_sync(self):
        """Kill the sandbox synchronously."""
        asyncio.run(self.kill())

    # Context manager support (E2B-compatible)
    def __enter__(self):
        """Synchronous context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Synchronous context manager exit."""
        self.kill_sync()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.kill()

    def set_timeout(self, timeout: int) -> None:
        """
        Set sandbox timeout (E2B-compatible).

        Changes the sandbox timeout during runtime.

        Args:
            timeout: New timeout in seconds
        """
        asyncio.run(self._set_timeout_async(timeout))

    async def _set_timeout_async(self, timeout: int) -> None:
        """Set timeout asynchronously via SDK."""
        await self._sdk.set_timeout(self.workspace_id, timeout)
        self.config.default_timeout_sec = timeout

    def get_info(self) -> SandboxInfo:
        """
        Get sandbox information (E2B-compatible).

        Returns:
            SandboxInfo object with sandbox details
        """
        return asyncio.run(self._get_info_async())

    async def _get_info_async(self) -> SandboxInfo:
        """Get sandbox info asynchronously via SDK."""
        return await self._sdk.get_sandbox_info(self.workspace_id)

    def pause(self) -> PausedWorkspace:
        """
        Pause the sandbox and save its state (E2B-compatible).

        Saves the workspace filesystem to a backup directory via remote API.

        Returns:
            PausedWorkspace object with backup information
        """
        return asyncio.run(self._sdk.pause_workspace(self.workspace_id))

    async def pause_async(self) -> PausedWorkspace:
        """
        Pause the sandbox asynchronously via remote API.

        Returns:
            PausedWorkspace object with backup information
        """
        return await self._sdk.pause_workspace(self.workspace_id)

    def resume(self, workspace_id: Optional[str] = None) -> "RemoteSandbox":
        """
        Resume a paused sandbox (E2B-compatible).

        Restores the workspace from its saved state via remote API.

        Args:
            workspace_id: Optional workspace ID to resume

        Returns:
            RemoteSandbox instance with restored workspace
        """
        return asyncio.run(self.resume_async(workspace_id))

    async def resume_async(self, workspace_id: Optional[str] = None) -> "RemoteSandbox":
        """
        Resume a paused sandbox asynchronously via remote API.

        Args:
            workspace_id: Optional workspace ID to resume

        Returns:
            RemoteSandbox instance with restored workspace
        """
        target_workspace_id = workspace_id or self.workspace_id

        # Resume the workspace via SDK
        workspace = await self._sdk.resume_workspace(target_workspace_id)
        logger.info(f"Remote sandbox resumed: {target_workspace_id}")

        # Update current sandbox
        self.workspace = workspace
        return self

    def get_metrics(self) -> List[SandboxMetrics]:
        """
        Get sandbox metrics (E2B-compatible).

        Returns CPU and memory metrics for the sandbox.

        Returns:
            List of SandboxMetrics objects with timestamped metrics
        """
        return asyncio.run(self._get_metrics_async())

    async def _get_metrics_async(self) -> List[SandboxMetrics]:
        """
        Get sandbox metrics asynchronously via remote API.

        Returns:
            List of SandboxMetrics objects with timestamped metrics
        """
        return await self._sdk.get_metrics(self.workspace_id)


class Sandbox:
    """
    E2B-compatible Sandbox API for ds-sandbox.

    This class provides a simple interface similar to E2B's Sandbox class.

    Usage:
        >>> from ds_sandbox import Sandbox
        >>>
        >>> sandbox = await Sandbox.create()
        >>>
        >>> # Upload a file
        >>> await sandbox.files.write("/workspace/data.csv", "col1,col2\\n1,2")
        >>>
        >>> # Run code
        >>> result = sandbox.run_code("import pandas as pd\\nprint('Hello')")
        >>>
        >>> # Or async
        >>> result = await sandbox.run_code_async("import pandas as pd\\nprint('Hello')")
        >>>
        >>> # Clean up
        >>> await sandbox.kill()
    """

    _instances: Dict[str, "Sandbox"] = {}

    def __init__(
        self,
        workspace_id: str,
        manager: SandboxManager,
        workspace: Any,
        config: Optional[SandboxConfig] = None,
        template: Optional[Template] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize sandbox (use Sandbox.create() instead)."""
        self.workspace_id = workspace_id
        self._manager = manager
        self.workspace = workspace
        self.config = config or SandboxConfig()
        self._files = Files(self)
        self._commands = Commands(self)
        self.sandbox_id = workspace_id

        # Template configuration
        self._template = template

        # Store metadata
        self._metadata = metadata or {}

        # Set user and workdir from template or config
        if template and template.user:
            self._user = template.user
        else:
            self._user = self.config.default_user

        if template and template.workdir:
            self._workdir = template.workdir
        else:
            self._workdir = self.config.default_workdir

    @classmethod
    def create(
        cls,
        timeout: int = 3600,
        envs: Optional[Dict[str, str]] = None,
        config: Optional[SandboxConfig] = None,
        workspace_id: Optional[str] = None,
        template: Optional[Template] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Sandbox":
        """
        Create a new sandbox instance synchronously (E2B-compatible).

        Usage:
            with Sandbox.create() as sandbox:
                result = sandbox.run_code("print('hello')")

            # Using a template
            from ds_sandbox import Template
            template = Template(
                id="my-template",
                env={"MY_VAR": "value"},
                user="customuser",
                workdir="/home/customuser",
            )
            with Sandbox.create(template=template) as sandbox:
                result = sandbox.run_code("print('hello')")

            # With metadata
            sandbox = Sandbox.create(metadata={"team": "data-science"})

        Args:
            timeout: Default timeout for code execution in seconds
            envs: Environment variables to set
            config: Sandbox configuration
            workspace_id: Optional workspace ID (auto-generated if not provided)
            template: Optional template to use for sandbox configuration
            metadata: Custom metadata for the sandbox

        Returns:
            Sandbox instance
        """
        return asyncio.run(cls.create_async(
            timeout=timeout,
            envs=envs,
            config=config,
            workspace_id=workspace_id,
            template=template,
            metadata=metadata,
        ))

    @classmethod
    async def create_async(
        cls,
        timeout: int = 3600,
        envs: Optional[Dict[str, str]] = None,
        config: Optional[SandboxConfig] = None,
        workspace_id: Optional[str] = None,
        external_workspace_path: Optional[str] = None,
        template: Optional[Template] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Sandbox":
        """
        Create a new sandbox instance (E2B-compatible).

        Args:
            timeout: Default timeout for code execution in seconds
            envs: Environment variables to set
            config: Sandbox configuration
            workspace_id: Optional workspace ID (auto-generated if not provided)
            external_workspace_path: For local mode, use this path directly (will create symlink)
            template: Optional template to use for sandbox configuration
            metadata: Custom metadata for the sandbox

        Returns:
            Sandbox instance
        """
        if config is None:
            config = SandboxConfig()

        # Generate workspace ID if not provided
        if workspace_id is None:
            workspace_id = f"ws-{uuid.uuid4().hex[:12]}"

        # If api_endpoint is configured, use remote sandbox
        if config.api_endpoint:
            # Create SDK for remote API
            sdk = SandboxSDK(api_endpoint=config.api_endpoint)

            # Create workspace on remote server with metadata
            workspace = await sdk.create_workspace(workspace_id, metadata=metadata)
            logger.info(f"Created remote sandbox with workspace: {workspace_id} at {config.api_endpoint}")

            # Create remote sandbox instance
            sandbox = RemoteSandbox(
                workspace_id=workspace_id,
                sdk=sdk,
                workspace=workspace,
                config=config,
            )
            cls._instances[workspace_id] = sandbox
            return sandbox

        # Create manager for local execution
        manager = SandboxManager(config=config)

        # For local mode with external workspace, pass external_path to manager
        # Manager will create symlink instead of real directory
        if external_workspace_path and config.default_backend == "local":
            try:
                workspace = await manager.create_workspace(workspace_id, external_path=external_workspace_path, metadata=metadata)
                logger.info(f"Created workspace with symlink: {workspace_id} -> {external_workspace_path}")
            except FileExistsError:
                workspace = await manager.get_workspace(workspace_id)
                logger.info(f"Using existing workspace: {workspace_id}")
        else:
            try:
                # Create workspace
                workspace = await manager.create_workspace(workspace_id, metadata=metadata)
                logger.info(f"Created sandbox with workspace: {workspace_id}")
            except FileExistsError:
                # Workspace already exists, get it
                workspace = await manager.get_workspace(workspace_id)
                logger.info(f"Using existing workspace: {workspace_id}")

        # Create sandbox instance
        sandbox = cls(
            workspace_id=workspace_id,
            manager=manager,
            workspace=workspace,
            config=config,
            template=template,
            metadata=metadata,
        )

        # Apply template files and environment variables
        if template:
            # Create files defined in template
            for file_path, content in template.files.items():
                try:
                    sandbox.files.write(file_path, content)
                except Exception as e:
                    logger.warning(f"Failed to create template file {file_path}: {e}")

            # Merge template environment variables
            if template.env:
                envs = envs or {}
                envs.update(template.env)
                sandbox._envs = envs

        # Store environment variables
        if envs:
            sandbox._envs = envs

        # Store in instances
        cls._instances[workspace_id] = sandbox

        return sandbox

    @property
    def files(self) -> Files:
        """File operations (E2B-compatible)."""
        return self._files

    @property
    def commands(self) -> Commands:
        """Command execution (E2B-compatible)."""
        return self._commands

    @property
    def user(self) -> str:
        """
        Get the current user (E2B-compatible).

        Returns:
            The current user name
        """
        return self._user

    @property
    def workdir(self) -> str:
        """
        Get the current working directory (E2B-compatible).

        Returns:
            The current working directory path
        """
        return self._workdir

    def run_code(
        self,
        code: str,
        timeout: Optional[int] = None,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> CodeResult:
        """
        Run Python code synchronously (E2B-compatible).

        Args:
            code: Python code to execute
            timeout: Timeout in seconds
            on_stdout: Callback for stdout stream data
            on_stderr: Callback for stderr stream data
            on_result: Callback for result data (charts, tables, etc.)

        Returns:
            CodeResult with stdout, stderr, error, etc.
        """
        return asyncio.run(self.run_code_async(
            code,
            timeout,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            on_result=on_result,
        ))

    async def run_code_async(
        self,
        code: str,
        timeout: Optional[int] = None,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> CodeResult:
        """
        Run Python code asynchronously (E2B-compatible).

        Args:
            code: Python code to execute
            timeout: Timeout in seconds
            on_stdout: Callback for stdout stream data
            on_stderr: Callback for stderr stream data
            on_result: Callback for result data (charts, tables, etc.)

        Returns:
            CodeResult with stdout, stderr, error, etc.
        """
        effective_timeout = timeout or self.config.default_timeout_sec

        # Try Jupyter execution for chart support
        # Check if the local backend supports Jupyter
        use_jupyter = getattr(self.config, 'use_jupyter', False)

        if use_jupyter:
            try:
                # Get the local backend and use Jupyter execution
                backend = self._manager.get_backend("local")
                if hasattr(backend, 'execute_jupyter'):
                    result = await backend.execute_jupyter(
                        ExecutionRequest(
                            code=code,
                            workspace_id=self.workspace_id,
                            timeout_sec=effective_timeout,
                            memory_mb=self.config.default_memory_mb,
                        ),
                        self.workspace,
                        on_stdout=on_stdout,
                        on_stderr=on_stderr,
                        on_result=on_result,
                    )

                    # Convert results with charts
                    results = []
                    metadata = result.metadata or {}
                    jupyter_results = metadata.get('results', [])

                    for r in jupyter_results:
                        results.append(r)  # Contains png, text, etc.

                    error = metadata.get('error')
                    if error:
                        error = {
                            "name": error.get('name', 'Error'),
                            "value": error.get('value', ''),
                            "traceback": error.get('traceback', ''),
                        }

                    return CodeResult(
                        success=result.success,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        error=error,
                        logs=ExecutionLogs(stdout=result.stdout, stderr=result.stderr),
                        results=results,
                    )
            except Exception as e:
                # Fall back to regular execution
                pass

        # Regular execution (no chart support)
        request = ExecutionRequest(
            code=code,
            workspace_id=self.workspace_id,
            timeout_sec=effective_timeout,
            memory_mb=self.config.default_memory_mb,
            env_vars={} if not hasattr(self, '_envs') else self._envs,
        )

        # Execute
        result = await self._manager.execute(request)

        # Call stdout/stderr callbacks if provided (for regular execution)
        if on_stdout and result.stdout:
            on_stdout(result.stdout)
        if on_stderr and result.stderr:
            on_stderr(result.stderr)

        # Convert to E2B-compatible format
        error = None
        if not result.success:
            error = {
                "name": result.stderr.split("\n")[0] if result.stderr else "Error",
                "value": result.stderr,
                "traceback": result.stderr,
            }

        return CodeResult(
            success=result.success,
            stdout=result.stdout,
            stderr=result.stderr,
            error=error,
            logs=ExecutionLogs(stdout=result.stdout, stderr=result.stderr),
            results=[{"artifacts": result.artifacts}],
        )

    async def kill(self) -> None:
        """
        Kill the sandbox (E2B-compatible).

        This stops the sandbox and optionally cleans up the workspace.
        """
        if self.workspace_id in self._instances:
            del self._instances[self.workspace_id]

        # Delete workspace
        try:
            await self._manager.delete_workspace(self.workspace_id, force=True)
            logger.info(f"Killed sandbox: {self.workspace_id}")
        except Exception as e:
            logger.warning(f"Error deleting workspace: {e}")

    def set_timeout(self, timeout: int) -> None:
        """
        Set sandbox timeout (E2B-compatible).

        Changes the sandbox timeout during runtime.
        For local sandbox, this updates the config timeout value.

        Args:
            timeout: New timeout in seconds
        """
        self.config.default_timeout_sec = timeout
        logger.info(f"Set sandbox timeout to {timeout}s")

    def get_info(self) -> SandboxInfo:
        """
        Get sandbox information (E2B-compatible).

        Returns:
            SandboxInfo object with sandbox details
        """
        # Get workspace info from the workspace object
        workspace = self.workspace

        # Determine started_at from workspace
        started_at = workspace.created_at if hasattr(workspace, 'created_at') else ""
        last_used = workspace.last_used_at if hasattr(workspace, 'last_used_at') else None

        # Build metadata - merge sandbox metadata with workspace metadata
        info_metadata = {
            "workspace_id": self.workspace_id,
            "host_path": workspace.host_path if hasattr(workspace, 'host_path') else "",
            "guest_path": workspace.guest_path if hasattr(workspace, 'guest_path') else "/workspace",
            "status": workspace.status if hasattr(workspace, 'status') else "ready",
            "last_used_at": last_used,
        }
        # Add custom metadata (from workspace or sandbox)
        if hasattr(workspace, 'metadata') and workspace.metadata:
            info_metadata.update(workspace.metadata)
        elif self._metadata:
            info_metadata.update(self._metadata)

        return SandboxInfo(
            sandbox_id=self.workspace_id,
            template_id=None,
            name=self.workspace_id,
            metadata=info_metadata,
            started_at=started_at,
            end_at=None,
        )

    def pause(self) -> PausedWorkspace:
        """
        Pause the sandbox and save its state (E2B-compatible).

        Saves the workspace filesystem to a backup directory. The sandbox
        can be resumed later to restore its state.

        Returns:
            PausedWorkspace object with backup information
        """
        return asyncio.run(self.pause_async())

    async def pause_async(self) -> PausedWorkspace:
        """
        Pause the sandbox asynchronously (E2B-compatible).

        Saves the workspace filesystem to a backup directory.

        Returns:
            PausedWorkspace object with backup information
        """
        paused_workspace = await self._manager.pause_workspace(self.workspace_id)
        logger.info(f"Sandbox paused: {self.workspace_id}")
        return paused_workspace

    def resume(self, workspace_id: Optional[str] = None) -> "Sandbox":
        """
        Resume a paused sandbox (E2B-compatible).

        Restores the workspace from its saved state. If workspace_id is provided,
        it resumes that workspace; otherwise, resumes the current sandbox.

        Args:
            workspace_id: Optional workspace ID to resume (defaults to current sandbox's workspace_id)

        Returns:
            Sandbox instance with restored workspace
        """
        return asyncio.run(self.resume_async(workspace_id))

    async def resume_async(self, workspace_id: Optional[str] = None) -> "Sandbox":
        """
        Resume a paused sandbox asynchronously (E2B-compatible).

        Restores the workspace from its saved state.

        Args:
            workspace_id: Optional workspace ID to resume (defaults to current sandbox's workspace_id)

        Returns:
            Sandbox instance with restored workspace
        """
        target_workspace_id = workspace_id or self.workspace_id

        # Resume the workspace
        workspace = await self._manager.resume_workspace(target_workspace_id)
        logger.info(f"Sandbox resumed: {target_workspace_id}")

        # Update the current sandbox's workspace
        self.workspace = workspace
        return self

    def get_metrics(self) -> List[SandboxMetrics]:
        """
        Get sandbox metrics (E2B-compatible).

        Returns CPU and memory metrics for the sandbox.

        Returns:
            List of SandboxMetrics objects with timestamped metrics
        """
        return asyncio.run(self._get_metrics_async())

    async def _get_metrics_async(self) -> List[SandboxMetrics]:
        """
        Get sandbox metrics asynchronously.

        Returns CPU and memory metrics for the workspace.

        Returns:
            List of SandboxMetrics objects with timestamped metrics
        """
        # Collect current metrics
        current_metrics = self._manager.collect_workspace_metrics(self.workspace_id)

        # Get metrics history
        metrics_history = self._manager.get_workspace_metrics(self.workspace_id)

        # Ensure current metrics is in the history
        if not metrics_history or metrics_history[-1].timestamp != current_metrics.timestamp:
            metrics_history.append(current_metrics)

        return metrics_history

    # Context manager support (E2B-compatible)
    def __enter__(self):
        """Synchronous context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Synchronous context manager exit."""
        asyncio.run(self.kill())

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.kill()

    @classmethod
    def get(cls, sandbox_id: str) -> Optional["Sandbox"]:
        """Get an existing sandbox by ID."""
        return cls._instances.get(sandbox_id)

    @classmethod
    def list(cls, state: Optional[str] = None) -> List[str]:
        """
        List all active sandbox IDs.

        Args:
            state: Optional filter by state ("running" or "paused")

        Returns:
            List of sandbox IDs
        """
        if state:
            # Filter based on state
            return list(cls._instances.keys())
        return list(cls._instances.keys())

    @classmethod
    async def list_paused_workspaces_async(cls) -> List[PausedWorkspace]:
        """
        List all paused workspaces.

        Returns:
            List of PausedWorkspace objects
        """
        # Get manager from any existing instance
        if cls._instances:
            sandbox = next(iter(cls._instances.values()))
            if hasattr(sandbox, '_manager'):
                return await sandbox._manager.list_paused_workspaces()
        return []

    @classmethod
    def list_paused_workspaces(cls) -> List[PausedWorkspace]:
        """
        List all paused workspaces synchronously.

        Returns:
            List of PausedWorkspace objects
        """
        return asyncio.run(cls.list_paused_workspaces_async())
