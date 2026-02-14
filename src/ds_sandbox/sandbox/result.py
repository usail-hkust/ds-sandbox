"""
Result classes for code execution (E2B-compatible).
"""

from typing import Any, Dict, List, Optional


class ExecutionLogs:
    """Execution logs (E2B-compatible)."""

    def __init__(self, stdout: str = "", stderr: str = ""):
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
        logs: Optional[ExecutionLogs] = None,
        results: Optional[List[Dict[str, Any]]] = None,
    ):
        self.success = success
        self.error = error
        self._stdout = stdout
        self._stderr = stderr
        self.logs = logs or ExecutionLogs(stdout=stdout, stderr=stderr)
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
