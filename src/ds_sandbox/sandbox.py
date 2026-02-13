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
from ds_sandbox.types import ExecutionRequest

logger = logging.getLogger(__name__)


class Files:
    """File operations handler (E2B-compatible)."""

    def __init__(self, sandbox: "Sandbox"):
        """Initialize with parent sandbox."""
        self._sandbox = sandbox

    def write(self, path: str, content: Union[bytes, str]) -> str:
        """Write content to a file synchronously (E2B-compatible).

        Args:
            path: Destination path in the sandbox.
            content: File content as bytes or string

        Returns:
            The path where the file was saved
        """
        return asyncio.run(self._write_async(path, content))

    async def _write_async(self, path: str, content: Union[bytes, str]) -> str:
        """Write content to a file asynchronously.

        Args:
            path: Destination path in the sandbox.
                  Examples:
                    - "/home/user/data.csv" -> {workspace}/home/user/data.csv
                    - "data.csv" -> {workspace}/data.csv
            content: File content as bytes or string

        Returns:
            The path where the file was saved
        """
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        # 直接将用户传入的路径拼接到 workspace 目录下
        # /home/user/data.csv -> {workspace}/home/user/data.csv
        # data.csv -> {workspace}/data.csv
        rel_path = path.lstrip("/")

        # 构建实际的主机路径
        host_path = Path(workspace.host_path) / rel_path

        # 创建父目录（如果不存在）
        host_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入内容
        if isinstance(content, str):
            host_path.write_text(content, encoding="utf-8")
        else:
            host_path.write_bytes(content)

        logger.debug(f"Wrote file to {host_path}")
        return path

    def read(self, path: str) -> bytes:
        """Read content from a file synchronously (E2B-compatible).

        Args:
            path: Path to the file in the sandbox.

        Returns:
            File content as bytes
        """
        return asyncio.run(self._read_async(path))

    async def _read_async(self, path: str) -> bytes:
        """Read content from a file in the sandbox asynchronously.

        Args:
            path: Path to the file in the sandbox.

        Returns:
            File content as bytes
        """
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        # 直接拼接到 workspace 目录
        rel_path = path.lstrip("/")
        host_path = Path(workspace.host_path) / rel_path

        if not host_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return host_path.read_bytes()

    async def list(self, path: str = None) -> List[Dict[str, Any]]:
        """List files in a directory.

        Args:
            path: Directory path in the sandbox (default: workspace root)

        Returns:
            List of file/directory information
        """
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        # 默认列出 workspace 根目录
        if path is None:
            host_path = Path(workspace.host_path)
        else:
            rel_path = path.lstrip("/")
            host_path = Path(workspace.host_path) / rel_path

        if not host_path.exists():
            return []

        results = []
        for item in host_path.iterdir():
            results.append({
                "name": item.name,
                "path": str(item),
                "is_file": item.is_file(),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else 0,
            })

        return results


class Commands:
    """Command execution handler (E2B-compatible)."""

    def __init__(self, sandbox: "Sandbox"):
        """Initialize with parent sandbox."""
        self._sandbox = sandbox

    def run(self, cmd: str, timeout: int = 60) -> "CommandResult":
        """Run a shell command in the sandbox.

        Args:
            cmd: Command to execute
            timeout: Timeout in seconds

        Returns:
            CommandResult with stdout, stderr, exit_code
        """
        return asyncio.run(self._run_async(cmd, timeout))

    async def _run_async(self, cmd: str, timeout: int) -> "CommandResult":
        """Run command asynchronously."""
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        import subprocess
        import sys

        try:
            # 使用系统 Python 而不是 shell 命令
            # 这样可以确保命令正确执行
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workspace.host_path,
                env={**__import__('os').environ, 'PATH': __import__('os').environ.get('PATH', '')},
            )
            return CommandResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired as e:
            return CommandResult(
                exit_code=-1,
                stdout=e.stdout or "",
                stderr=f"Command timed out after {timeout}s",
            )


class CommandResult:
    """Result of a command execution (E2B-compatible)."""

    def __init__(
        self,
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
    ):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class CodeResult:
    """Result of code execution (E2B-compatible)."""

    def __init__(
        self,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        error: Optional[Dict[str, Any]] = None,
        logs: Optional["ExecutionLogs2"] = None,
        results: Optional[List[Dict[str, Any]]] = None,
    ):
        self.success = success
        self.error = error
        self._stdout = stdout
        self._stderr = stderr
        self.logs = logs or ExecutionLogs2(stdout=stdout, stderr=stderr)
        self.results = results or []

    @property
    def stdout(self) -> str:
        """Get stdout (E2B-compatible)."""
        return self._stdout

    @property
    def stderr(self) -> str:
        """Get stderr (E2B-compatible)."""
        return self._stderr

    @property
    def text(self) -> str:
        """Get the stdout as text (E2B-compatible)."""
        return self._stdout


class ExecutionLogs2:
    """Execution logs (E2B-compatible)."""

    def __init__(self, stdout: str = "", stderr: str = ""):
        self.stdout = stdout
        self.stderr = stderr


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
    ):
        """Initialize sandbox (use Sandbox.create() instead)."""
        self.workspace_id = workspace_id
        self._manager = manager
        self.workspace = workspace
        self.config = config or SandboxConfig()
        self._files = Files(self)
        self._commands = Commands(self)
        self.sandbox_id = workspace_id

    @classmethod
    def create(
        cls,
        timeout: int = 3600,
        envs: Optional[Dict[str, str]] = None,
        config: Optional[SandboxConfig] = None,
        workspace_id: Optional[str] = None,
    ) -> "Sandbox":
        """
        Create a new sandbox instance synchronously (E2B-compatible).

        Usage:
            with Sandbox.create() as sandbox:
                result = sandbox.run_code("print('hello')")

        Args:
            timeout: Default timeout for code execution in seconds
            envs: Environment variables to set
            config: Sandbox configuration
            workspace_id: Optional workspace ID (auto-generated if not provided)

        Returns:
            Sandbox instance
        """
        return asyncio.run(cls.create_async(
            timeout=timeout,
            envs=envs,
            config=config,
            workspace_id=workspace_id,
        ))

    @classmethod
    async def create_async(
        cls,
        timeout: int = 3600,
        envs: Optional[Dict[str, str]] = None,
        config: Optional[SandboxConfig] = None,
        workspace_id: Optional[str] = None,
        external_workspace_path: Optional[str] = None,
    ) -> "Sandbox":
        """
        Create a new sandbox instance (E2B-compatible).

        Args:
            timeout: Default timeout for code execution in seconds
            envs: Environment variables to set
            config: Sandbox configuration
            workspace_id: Optional workspace ID (auto-generated if not provided)
            external_workspace_path: For local mode, use this path directly (will create symlink)

        Returns:
            Sandbox instance
        """
        if config is None:
            config = SandboxConfig()

        # Create manager
        manager = SandboxManager(config=config)

        # Generate workspace ID if not provided
        if workspace_id is None:
            workspace_id = f"ws-{uuid.uuid4().hex[:12]}"

        # For local mode with external workspace, pass external_path to manager
        # Manager will create symlink instead of real directory
        if external_workspace_path and config.default_backend == "local":
            try:
                workspace = await manager.create_workspace(workspace_id, external_path=external_workspace_path)
                logger.info(f"Created workspace with symlink: {workspace_id} -> {external_workspace_path}")
            except FileExistsError:
                workspace = await manager.get_workspace(workspace_id)
                logger.info(f"Using existing workspace: {workspace_id}")
        else:
            try:
                # Create workspace
                workspace = await manager.create_workspace(workspace_id)
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
        )

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
                        logs=ExecutionLogs2(stdout=result.stdout, stderr=result.stderr),
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
            logs=ExecutionLogs2(stdout=result.stdout, stderr=result.stderr),
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
    def list(cls) -> List[str]:
        """List all active sandbox IDs."""
        return list(cls._instances.keys())
