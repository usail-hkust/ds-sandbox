"""
Command execution handler (E2B-compatible).
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


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

    def run_streaming(
        self,
        cmd: str,
        timeout: int = 60,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
    ) -> "CommandResult":
        """Run a command with streaming output.

        Args:
            cmd: Command to execute
            timeout: Timeout in seconds
            on_stdout: Callback for each stdout line
            on_stderr: Callback for each stderr line

        Returns:
            CommandResult with final stdout/stderr
        """
        return asyncio.run(self._run_streaming_async(cmd, timeout, on_stdout, on_stderr))

    async def _run_streaming_async(
        self,
        cmd: str,
        timeout: int,
        on_stdout: Optional[Callable[[str], None]],
        on_stderr: Optional[Callable[[str], None]],
    ) -> "CommandResult":
        """Run command with streaming output asynchronously."""
        import subprocess
        import threading

        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        stdout_lines = []
        stderr_lines = []
        stdout_lock = threading.Lock()
        stderr_lock = threading.Lock()
        process = None

        def read_stdout(proc):
            """Read from stdout in a separate thread."""
            if proc.stdout:
                for line in iter(proc.stdout.readline, ''):
                    if line:
                        line = line.rstrip('\n')
                        with stdout_lock:
                            stdout_lines.append(line)
                        if on_stdout:
                            on_stdout(line)

        def read_stderr(proc):
            """Read from stderr in a separate thread."""
            if proc.stderr:
                for line in iter(proc.stderr.readline, ''):
                    if line:
                        line = line.rstrip('\n')
                        with stderr_lock:
                            stderr_lines.append(line)
                        if on_stderr:
                            on_stderr(line)

        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=workspace.host_path,
                env={**__import__('os').environ, 'PATH': __import__('os').environ.get('PATH', '')},
            )

            # Start threads to read stdout and stderr
            stdout_thread = threading.Thread(target=read_stdout, args=(process,), daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, args=(process,), daemon=True)

            stdout_thread.start()
            stderr_thread.start()

            # Wait for process to complete with timeout
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return CommandResult(
                    exit_code=-1,
                    stdout='\n'.join(stdout_lines),
                    stderr=f"Command timed out after {timeout}s",
                )

            # Wait for reader threads to finish
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)

            return CommandResult(
                exit_code=return_code,
                stdout='\n'.join(stdout_lines),
                stderr='\n'.join(stderr_lines),
            )

        except Exception as e:
            return CommandResult(
                exit_code=-1,
                stdout='\n'.join(stdout_lines),
                stderr=f"Error running command: {str(e)}",
            )

    def start(
        self,
        cmd: str,
        env: Optional[Dict[str, str]] = None,
    ) -> "Process":
        """Start a command in the background.

        Args:
            cmd: Command to execute
            env: Environment variables

        Returns:
            Process object with pid and methods to interact
        """
        return asyncio.run(self._start_async(cmd, env))

    async def _start_async(
        self,
        cmd: str,
        env: Optional[Dict[str, str]] = None,
    ) -> "Process":
        """Start a command in the background asynchronously.

        Args:
            cmd: Command to execute
            env: Environment variables

        Returns:
            Process object with pid and methods to interact
        """
        import subprocess
        import os

        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        # Build environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        process_env['PATH'] = os.environ.get('PATH', '')

        # Start the process
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=workspace.host_path,
            env=process_env,
            start_new_session=True,  # Detach process from parent
        )

        logger.debug(f"Started background process with PID: {process.pid}")

        return Process(
            pid=process.pid,
            process=process,
            workspace_path=workspace.host_path,
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


class Process:
    """A running process in the sandbox."""

    def __init__(self, pid: int, process: Any, workspace_path: str):
        """Initialize the Process object.

        Args:
            pid: Process ID
            process: The subprocess.Popen object
            workspace_path: Path to the workspace directory
        """
        self.pid = pid
        self._process = process
        self._workspace_path = workspace_path
        self._exit_code: Optional[int] = None

    def is_running(self) -> bool:
        """Check if process is still running.

        Returns:
            True if the process is still running, False otherwise
        """
        if self._process is None:
            return False
        return self._process.poll() is None

    def wait(self, timeout: Optional[int] = None) -> int:
        """Wait for process to complete, return exit code.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Exit code of the process

        Raises:
            subprocess.TimeoutExpired: If timeout is exceeded
        """
        if self._process is None:
            return self._exit_code if self._exit_code is not None else -1

        self._exit_code = self._process.wait(timeout=timeout)
        return self._exit_code

    def kill(self) -> None:
        """Kill the process."""
        if self._process is not None and self.is_running():
            self._process.kill()
            self._process.wait()
            self._exit_code = self._process.returncode

    @property
    def exit_code(self) -> Optional[int]:
        """Get exit code if process has completed.

        Returns:
            Exit code if process has completed, None otherwise
        """
        if self._process is None:
            return self._exit_code
        if self.is_running():
            return None
        return self._process.returncode


# Import Sandbox type hint (lazy import to avoid circular imports)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ds_sandbox.sandbox.sandbox import Sandbox
