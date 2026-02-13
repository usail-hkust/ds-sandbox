"""
Unit tests for local subprocess backend.
"""

from pathlib import Path

import pytest

from ds_sandbox.backends.base import SandboxBackend
from ds_sandbox.backends.local import LocalSubprocessSandbox
from ds_sandbox.types import ExecutionRequest, Workspace


def _create_workspace(tmp_path: Path) -> Workspace:
    workspace_path = tmp_path / "workspace-local"
    workspace_path.mkdir(parents=True, exist_ok=True)
    (workspace_path / "data").mkdir(exist_ok=True)
    (workspace_path / "models").mkdir(exist_ok=True)
    (workspace_path / "outputs").mkdir(exist_ok=True)

    return Workspace(
        workspace_id="workspace-local",
        host_path=str(workspace_path),
        created_at="2026-02-13T00:00:00Z",
    )


class TestLocalSubprocessSandboxInit:
    def test_default_init(self):
        backend = LocalSubprocessSandbox()

        assert backend.default_timeout_sec == 3600
        assert backend.python_executable

    def test_custom_init(self):
        backend = LocalSubprocessSandbox(
            config={
                "python_executable": "/usr/bin/python3",
                "timeout_sec": 120,
            }
        )

        assert backend.python_executable == "/usr/bin/python3"
        assert backend.default_timeout_sec == 120

    def test_is_sandbox_backend(self):
        backend = LocalSubprocessSandbox()
        assert isinstance(backend, SandboxBackend)


class TestLocalSubprocessSandboxExecute:
    @pytest.mark.asyncio
    async def test_execute_success(self, tmp_path: Path):
        backend = LocalSubprocessSandbox()
        workspace = _create_workspace(tmp_path)

        request = ExecutionRequest(
            code=(
                "from pathlib import Path\n"
                "Path('outputs/result.txt').write_text('ok', encoding='utf-8')\n"
                "print('hello-local')\n"
            ),
            workspace_id=workspace.workspace_id,
            env_vars={"LOCAL_BACKEND_TEST": "1"},
        )

        result = await backend.execute(request, workspace)

        assert result.success is True
        assert "hello-local" in result.stdout
        assert result.exit_code == 0
        assert result.backend == "local"
        assert "outputs/result.txt" in result.artifacts

    @pytest.mark.asyncio
    async def test_execute_timeout(self, tmp_path: Path):
        backend = LocalSubprocessSandbox()
        workspace = _create_workspace(tmp_path)

        request = ExecutionRequest(
            code="import time\ntime.sleep(2)\n",
            workspace_id=workspace.workspace_id,
            timeout_sec=1,
        )

        result = await backend.execute(request, workspace)

        assert result.success is False
        assert result.exit_code == -1
        assert "timed out" in result.stderr.lower()


class TestLocalSubprocessSandboxHealth:
    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        backend = LocalSubprocessSandbox()
        result = await backend.health_check()

        assert result["status"] == "healthy"
        assert result["backend"] == "local"
        assert "python_executable" in result

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        backend = LocalSubprocessSandbox(config={"python_executable": "__missing_python_binary__"})
        result = await backend.health_check()

        assert result["status"] == "unhealthy"
        assert result["backend"] == "local"


class TestLocalSubprocessSandboxCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_noop(self):
        backend = LocalSubprocessSandbox()
        await backend.cleanup("workspace-local")
