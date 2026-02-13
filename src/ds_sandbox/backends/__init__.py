"""
Sandbox backend base class

All sandbox backends must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Optional
from ds_sandbox.types import ExecutionRequest, ExecutionResult, Workspace


class SandboxBackend(ABC):
    """
    Abstract base class for sandbox backends.

    All backends (Docker, Firecracker, Kata) must inherit from this.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @abstractmethod
    async def execute(
        self,
        request: ExecutionRequest,
        workspace: Workspace
    ) -> ExecutionResult:
        """
        Execute code in isolated environment.

        Args:
            request: Execution request with all parameters
            workspace: Workspace information

        Returns:
            ExecutionResult with execution details
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict:
        """
        Check if backend is healthy and available.

        Returns:
            Dict with status information
        """
        pass

    @abstractmethod
    async def cleanup(self, workspace_id: str) -> None:
        """
        Cleanup resources for a workspace.

        Args:
            workspace_id: Workspace ID to cleanup
        """
        pass

    async def stop_execution(self, execution_id: str, workspace_id: str) -> bool:
        """
        Stop a running execution.

        Default implementation returns False (not supported).
        Override in backend if stopping is supported.

        Args:
            execution_id: Execution identifier
            workspace_id: Workspace identifier

        Returns:
            True if execution was stopped, False otherwise
        """
        return False
