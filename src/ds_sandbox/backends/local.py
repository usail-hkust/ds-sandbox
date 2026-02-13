"""
Local subprocess sandbox backend.

Executes Python code directly on the host in the workspace directory.
This backend is intended for local development and benchmarking only.

Supports two execution modes:
1. subprocess: Direct execution (fast, no chart support)
2. jupyter: Jupyter kernel execution (slower, supports automatic chart capture)
"""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ds_sandbox.backends.base import SandboxBackend
from ds_sandbox.types import ExecutionRequest, ExecutionResult, Workspace


class JupyterExecutor:
    """Jupyter-based code executor with automatic chart capture."""

    def __init__(self, python_executable: str = None):
        self.python_executable = python_executable or sys.executable
        self._km = None
        self._kc = None

    async def start(self):
        """Start a Jupyter kernel."""
        try:
            from jupyter_client import KernelManager
        except ImportError:
            raise ImportError("jupyter_client is required. Install with: pip install jupyter_client")

        self._km = KernelManager(kernel_name='python3', manager_class='process')
        self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()

        # Wait for kernel to be ready
        self._kc.wait_for_ready(timeout=30)

    async def execute(
        self,
        code: str,
        timeout: int = 60,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute code and capture output + charts.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds
            on_stdout: Optional callback for stdout stream data
            on_stderr: Optional callback for stderr stream data
            on_result: Optional callback for result data (charts, tables, etc.)

        Returns:
            dict with keys: stdout, stderr, results (list of {png, text, table})
        """
        if not self._kc:
            await self.start()

        # Execute the code
        self._kc.execute(code)

        stdout = ""
        stderr = ""
        results = []
        error = None

        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=timeout)
            except Exception:
                break

            # Handle both message objects and dicts
            if isinstance(msg, dict):
                msg_type = msg.get('msg_type', '')
                content = msg.get('content', {})
            else:
                msg_type = getattr(msg, 'msg_type', '')
                content = getattr(msg, 'content', {})

            if msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

            elif msg_type == 'stream':
                # stdout or stderr
                name = content.get('name', 'stdout')
                text = content.get('text', '')
                if name == 'stdout':
                    stdout += text
                    # Call stdout callback if provided
                    if on_stdout:
                        on_stdout(text)
                else:
                    stderr += text
                    # Call stderr callback if provided
                    if on_stderr:
                        on_stderr(text)

            elif msg_type == 'execute_result':
                # Return value displayed with IPython
                data = content.get('data', {})
                result_item = {}
                if 'text/plain' in data:
                    result_item['text'] = data['text/plain']
                    results.append(result_item)
                    # Call result callback if provided
                    if on_result:
                        on_result(result_item)
                if 'image/png' in data:
                    result_item = {'png': data['image/png']}
                    results.append(result_item)
                    # Call result callback if provided
                    if on_result:
                        on_result(result_item)

            elif msg_type == 'display_data':
                # Explicit display (like plt.show())
                data = content.get('data', {})
                if 'image/png' in data:
                    result_item = {'png': data['image/png']}
                    results.append(result_item)
                    # Call result callback if provided
                    if on_result:
                        on_result(result_item)

            elif msg_type == 'error':
                # Execution error
                error_name = content.get('ename', 'Error')
                error_value = content.get('evalue', '')
                error_traceback = content.get('traceback', '')
                error = {
                    'name': error_name,
                    'value': error_value,
                    'traceback': '\n'.join(error_traceback)
                }

        return {
            'stdout': stdout,
            'stderr': stderr,
            'results': results,
            'error': error,
        }

    async def stop(self):
        """Stop the Jupyter kernel."""
        if self._kc:
            self._kc.stop_channels()
        if self._km:
            self._km.shutdown_kernel()


class LocalSubprocessSandbox(SandboxBackend):
    """
    Local subprocess backend.

    Characteristics:
    - No container/VM isolation
    - Runs code under local Python interpreter
    - Uses workspace host path as execution cwd
    """

    DEFAULT_TIMEOUT_SEC = 3600

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.config = config or {}
        self.python_executable = self.config.get("python_executable", sys.executable or "python3")
        self.default_timeout_sec = int(self.config.get("timeout_sec", self.DEFAULT_TIMEOUT_SEC))

    async def execute(
        self,
        request: ExecutionRequest,
        workspace: Workspace,
    ) -> ExecutionResult:
        start_time = time.time()
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        workspace_id = workspace.workspace_id
        workspace_path = Path(workspace.host_path)
        script_path: Optional[Path] = None
        process: Optional[asyncio.subprocess.Process] = None

        if not workspace_path.exists():
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Workspace path does not exist: {workspace_path}",
                exit_code=-1,
                duration_ms=duration_ms,
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="local",
                isolation_level="process",
                metadata={"error": "workspace_not_found"},
            )

        timeout_sec = request.timeout_sec or self.default_timeout_sec

        try:
            script_path = await asyncio.to_thread(self._write_temp_script, workspace_path, request.code)
            env = self._build_env(
                request.env_vars,
                workspace,
                network_policy=request.network_policy,
                network_whitelist=request.network_whitelist,
                allow_internet=request.allow_internet
            )

            process = await asyncio.create_subprocess_exec(
                self.python_executable,
                str(script_path),
                cwd=str(workspace_path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = process.returncode if process.returncode is not None else -1
            duration_ms = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                success=exit_code == 0,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                duration_ms=duration_ms,
                artifacts=self._collect_artifacts(workspace_path),
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="local",
                isolation_level="process",
                metadata={
                    "python_executable": self.python_executable,
                    "cwd": str(workspace_path),
                    "pid": process.pid,
                    "network_policy": request.network_policy,
                    "allow_internet": request.allow_internet,
                    "enable_gpu": request.enable_gpu,
                },
            )

        except asyncio.TimeoutError:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()

            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {timeout_sec} seconds",
                exit_code=-1,
                duration_ms=duration_ms,
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="local",
                isolation_level="process",
                metadata={"error": "timeout"},
            )

        except FileNotFoundError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Python executable not found: {self.python_executable} ({e})",
                exit_code=-1,
                duration_ms=duration_ms,
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="local",
                isolation_level="process",
                metadata={"error": "python_not_found"},
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Unexpected error: {e}",
                exit_code=-1,
                duration_ms=duration_ms,
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="local",
                isolation_level="process",
                metadata={"error": str(e)},
            )
        finally:
            if script_path is not None:
                try:
                    script_path.unlink(missing_ok=True)
                except Exception:
                    pass

    async def health_check(self) -> Dict[str, Any]:
        executable = self._resolve_python_executable(self.python_executable)
        if executable is None:
            return {
                "status": "unhealthy",
                "backend": "local",
                "error": f"Python executable not found: {self.python_executable}",
            }

        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=5)
            version_output = stdout_bytes.decode("utf-8", errors="replace").strip() or stderr_bytes.decode(
                "utf-8", errors="replace"
            ).strip()

            if process.returncode != 0:
                return {
                    "status": "unhealthy",
                    "backend": "local",
                    "python_executable": executable,
                    "error": version_output or "python --version failed",
                }

            return {
                "status": "healthy",
                "backend": "local",
                "python_executable": executable,
                "python_version": version_output,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "local",
                "python_executable": executable,
                "error": str(e),
            }

    async def cleanup(self, workspace_id: str) -> None:
        # Local backend does not allocate per-workspace runtime resources.
        _ = workspace_id
        return None

    def _build_env(
        self,
        env_vars: Dict[str, str],
        workspace: Workspace,
        network_policy: str = "allow",
        network_whitelist: Optional[List[str]] = None,
        allow_internet: bool = True
    ) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "WORKSPACE": str(workspace.host_path),
                "WORKSPACE_DATA": str(Path(workspace.host_path) / "data"),
                "WORKSPACE_MODELS": str(Path(workspace.host_path) / "models"),
                "WORKSPACE_OUTPUTS": str(Path(workspace.host_path) / "outputs"),
            }
        )

        # Apply network configuration
        if not allow_internet:
            # Disable internet by setting invalid proxy
            env["HTTP_PROXY"] = ""
            env["HTTPS_PROXY"] = ""
            env["http_proxy"] = ""
            env["https_proxy"] = ""
            env["NO_PROXY"] = "*"
            env["no_proxy"] = "*"
        elif network_policy == "deny":
            # Deny network access
            env["HTTP_PROXY"] = ""
            env["HTTPS_PROXY"] = ""
            env["http_proxy"] = ""
            env["https_proxy"] = ""
            env["NO_PROXY"] = "*"
            env["no_proxy"] = "*"
        elif network_policy == "whitelist" and network_whitelist:
            # Set up whitelist via hosts file manipulation would be complex
            # For now, we'll use a wrapper script approach
            # The actual enforcement would require additional setup
            pass

        if env_vars:
            env.update(env_vars)
        return env

    def _write_temp_script(self, workspace_path: Path, code: str) -> Path:
        fd, temp_file = tempfile.mkstemp(
            prefix="_ds_sandbox_local_",
            suffix=".py",
            dir=str(workspace_path),
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(code)
        return Path(temp_file)

    async def execute_jupyter(
        self,
        request: ExecutionRequest,
        workspace: Workspace,
        on_stdout: Optional[Callable[[str], None]] = None,
        on_stderr: Optional[Callable[[str], None]] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> ExecutionResult:
        """Execute code using Jupyter kernel with automatic chart capture.

        This method supports matplotlib chart capture like E2B.

        Args:
            request: Execution request containing code and configuration
            workspace: Workspace to execute in
            on_stdout: Optional callback for stdout stream data
            on_stderr: Optional callback for stderr stream data
            on_result: Optional callback for result data (charts, tables, etc.)

        Returns:
            ExecutionResult with execution outcome
        """
        start_time = time.time()
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        workspace_id = workspace.workspace_id
        workspace_path = Path(workspace.host_path)

        if not workspace_path.exists():
            duration_ms = int((time.time() - start_time) * 0)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Workspace path does not exist: {workspace_path}",
                exit_code=-1,
                duration_ms=duration_ms,
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="local",
                isolation_level="jupyter",
                metadata={"error": "workspace_not_found"},
            )

        timeout_sec = request.timeout_sec or self.default_timeout_sec

        try:
            # Get or create Jupyter executor
            if not hasattr(self, '_jupyter') or self._jupyter is None:
                self._jupyter = JupyterExecutor(self.python_executable)
                await self._jupyter.start()

            # Execute code with streaming callbacks
            result = await self._jupyter.execute(
                request.code,
                timeout=timeout_sec,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
                on_result=on_result,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            # Convert results to E2B format
            # result['results'] is a list of {png: base64, text: string}
            results_formatted = []
            for r in result['results']:
                formatted = {}
                if 'png' in r:
                    formatted['png'] = r['png']  # base64 encoded PNG
                if 'text' in r:
                    formatted['text'] = r['text']
                if formatted:
                    results_formatted.append(formatted)

            error = result.get('error')
            success = error is None

            # Build error info if needed
            error_info = None
            if error:
                error_info = {
                    'name': error.get('name', 'Error'),
                    'value': error.get('value', ''),
                    'traceback': error.get('traceback', ''),
                }

            return ExecutionResult(
                success=success,
                stdout=result.get('stdout', ''),
                stderr=result.get('stderr', ''),
                exit_code=0 if success else 1,
                duration_ms=duration_ms,
                artifacts=self._collect_artifacts(workspace_path),
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="local",
                isolation_level="jupyter",
                metadata={
                    "python_executable": self.python_executable,
                    "cwd": str(workspace_path),
                    "execution_mode": "jupyter",
                    "results": results_formatted,
                    "error": error_info,
                },
            )

        except asyncio.TimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {timeout_sec} seconds",
                exit_code=-1,
                duration_ms=duration_ms,
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="local",
                isolation_level="jupyter",
                metadata={"error": "timeout"},
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Jupyter execution error: {e}",
                exit_code=-1,
                duration_ms=duration_ms,
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="local",
                isolation_level="jupyter",
                metadata={"error": str(e)},
            )

    async def shutdown_jupyter(self):
        """Shutdown the Jupyter kernel."""
        if hasattr(self, '_jupyter') and self._jupyter:
            await self._jupyter.stop()
            self._jupyter = None

    def _collect_artifacts(self, workspace_path: Path) -> list[str]:
        artifacts: list[str] = []
        outputs_dir = workspace_path / "outputs"
        if not outputs_dir.exists():
            return artifacts

        for artifact_path in outputs_dir.rglob("*"):
            if artifact_path.is_file():
                artifacts.append(str(artifact_path.relative_to(workspace_path)))

        return artifacts

    def _resolve_python_executable(self, executable: str) -> Optional[str]:
        if Path(executable).is_absolute():
            return executable if Path(executable).exists() else None
        return shutil.which(executable)
