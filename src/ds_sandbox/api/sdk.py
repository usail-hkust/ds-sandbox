"""
ds-sandbox Python SDK

A Python client library for interacting with the ds-sandbox API.
Supports both synchronous and asynchronous operations.

Example usage:
    import asyncio
    from ds_sandbox import SandboxSDK

    async def main():
        sdk = SandboxSDK(api_endpoint="http://localhost:8000")

        # Create workspace
        workspace = await sdk.create_workspace("my-workspace")

        # Execute code
        result = await sdk.execute(
            workspace_id="my-workspace",
            code="import pandas as pd; print(pd.__version__)"
        )

        print(result.stdout)

    asyncio.run(main())
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Literal, Optional

import aiohttp
from pydantic import BaseModel, Field

from ds_sandbox.errors import (
    SandboxError,
    ExecutionTimeoutError,
    ExecutionFailedError,
)
from ds_sandbox.types import ExecutionResult, Workspace, DatasetInfo

logger = logging.getLogger(__name__)

__all__ = ["SandboxSDK", "ExecutionStatus", "SDKConfig"]


class SDKConfig(BaseModel):
    """SDK configuration options"""
    api_endpoint: str = Field(default="http://localhost:8000", description="API endpoint URL")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries (seconds)")
    poll_interval: float = Field(default=0.5, description="Polling interval for execution status")


class ExecutionStatus(BaseModel):
    """Execution status response"""
    execution_id: str = Field(..., description="Execution ID")
    workspace_id: str = Field(..., description="Workspace ID")
    status: str = Field(..., description="Status: queued/running/completed/failed/stopped")
    backend: Optional[str] = Field(None, description="Backend used")
    created_at: str = Field(..., description="Creation timestamp")
    started_at: Optional[str] = Field(None, description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    stdout: Optional[str] = Field(None, description="Standard output (available after completion)")
    stderr: Optional[str] = Field(None, description="Standard error (available after completion)")
    exit_code: Optional[int] = Field(None, description="Exit code (available after completion)")
    duration_ms: Optional[int] = Field(None, description="Execution duration in milliseconds")


class SandboxSDK:
    """
    ds-sandbox Python SDK

    Provides a convenient interface to interact with the ds-sandbox API
    for workspace management and code execution.

    Attributes:
        endpoint: The API endpoint URL
        api_key: API key for authentication (optional)
        timeout: Request timeout in seconds

    Example:
        >>> sdk = SandboxSDK(api_endpoint="http://localhost:8000")
        >>> workspace = await sdk.create_workspace("my-workspace")
        >>> result = await sdk.execute("my-workspace", "print('Hello!')")
    """

    def __init__(
        self,
        api_endpoint: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        poll_interval: float = 0.5,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """
        Initialize the SDK.

        Args:
            api_endpoint: The API endpoint URL (default: http://localhost:8000)
            api_key: API key for authentication (optional)
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum retry attempts for failed requests (default: 3)
            retry_delay: Delay between retries in seconds (default: 1.0)
            poll_interval: Polling interval for execution status (default: 0.5)
            session: Optional aiohttp session to reuse
        """
        self.endpoint = api_endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.poll_interval = poll_interval
        self._session = session
        self._own_session = session is None

        logger.info(f"SandboxSDK initialized with endpoint: {self.endpoint}")

    @property
    def _default_headers(self) -> Dict[str, str]:
        """Generate default request headers"""
        headers = {
            "Content-Type": "application/json",
            "X-API-Version": "1.0.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._default_headers,
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying aiohttp session"""
        if self._own_session and self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.info("SandboxSDK session closed")

    async def __aenter__(self) -> "SandboxSDK":
        """Async context manager entry"""
        await self._get_session()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit"""
        await self.close()

    def _build_url(self, path: str) -> str:
        """Build full URL from path"""
        return f"{self.endpoint}{path}"

    async def _request(
        self,
        method: str,
        path: str,
        retry_on_error: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API path (e.g., "/v1/workspaces")
            retry_on_error: Whether to retry on failure
            **kwargs: Additional arguments for aiohttp

        Returns:
            JSON response as dictionary

        Raises:
            SandboxError: On API errors
        """
        url = self._build_url(path)
        session = await self._get_session()

        retries = 0
        last_error: Optional[Exception] = None

        while retries <= self.max_retries:
            try:
                logger.debug(f"{method} {url}")
                async with session.request(method, url, **kwargs) as response:
                    await self._check_response(response)

                    if response.content_type == "application/json":
                        return await response.json()
                    else:
                        return {"content": await response.text()}

            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(f"Request failed (attempt {retries + 1}): {e}")

                if retry_on_error and retries < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** retries))
                    retries += 1
                else:
                    break

        raise SandboxError(
            message=f"Request failed after {retries} attempts: {last_error}",
            error_code="REQUEST_FAILED",
        )

    async def _check_response(self, response: aiohttp.ClientResponse) -> None:
        """
        Check response for errors.

        Args:
            response: aiohttp response object

        Raises:
            SandboxError: On API errors
        """
        if response.status < 400:
            return

        try:
            error_data = await response.json()

            error_code = error_data.get("error_code", f"HTTP_{response.status}")
            message = error_data.get("message", response.reason)
            details = error_data.get("details", {})

            # Map to specific exception types
            # Note: We use SandboxError for all cases since the API response
            # doesn't include specific params (workspace_id, execution_id, etc.)
            # needed by specialized exceptions. The subclasses have different
            # signatures that don't accept message/error_code/details kwargs.
            raise SandboxError(message=message, error_code=error_code, details=details)

        except (ValueError, aiohttp.ClientError):
            # Fallback if we can't parse error response
            raise SandboxError(
                message=f"HTTP {response.status}: {response.reason}",
                error_code=f"HTTP_{response.status}",
            )

    # =========================================================================
    # Workspace Management
    # =========================================================================

    async def create_workspace(
        self,
        workspace_id: str,
        setup_dirs: Optional[List[str]] = None,
    ) -> Workspace:
        """
        Create a new workspace.

        Args:
            workspace_id: Unique identifier for the workspace
            setup_dirs: List of subdirectories to create (default: ["data", "models", "outputs"])

        Returns:
            Workspace object with details

        Example:
            >>> workspace = await sdk.create_workspace(
            ...     "my-experiment",
            ...     setup_dirs=["data", "models", "outputs"]
            ... )
            >>> print(workspace.host_path)
            /opt/workspaces/my-experiment
        """
        if setup_dirs is None:
            setup_dirs = ["data", "models", "outputs"]

        payload = {
            "workspace_id": workspace_id,
            "setup_dirs": setup_dirs,
        }

        logger.info(f"Creating workspace: {workspace_id}")
        data = await self._request("POST", "/v1/workspaces", json=payload)
        workspace = Workspace(**data)
        logger.info(f"Workspace created: {workspace_id}")
        return workspace

    async def get_workspace(self, workspace_id: str) -> Workspace:
        """
        Get workspace information.

        Args:
            workspace_id: Workspace identifier

        Returns:
            Workspace object with details

        Raises:
            WorkspaceNotFoundError: If workspace doesn't exist

        Example:
            >>> workspace = await sdk.get_workspace("my-experiment")
            >>> print(workspace.status)
            ready
        """
        logger.debug(f"Getting workspace: {workspace_id}")
        data = await self._request("GET", f"/v1/workspaces/{workspace_id}")
        return Workspace(**data)

    async def delete_workspace(self, workspace_id: str) -> None:
        """
        Delete a workspace and all its data.

        Args:
            workspace_id: Workspace identifier

        Raises:
            WorkspaceNotFoundError: If workspace doesn't exist

        Example:
            >>> await sdk.delete_workspace("my-experiment")
        """
        logger.info(f"Deleting workspace: {workspace_id}")
        await self._request("DELETE", f"/v1/workspaces/{workspace_id}")
        logger.info(f"Workspace deleted: {workspace_id}")

    async def list_workspaces(self) -> List[Workspace]:
        """
        List all workspaces.

        Returns:
            List of Workspace objects

        Example:
            >>> workspaces = await sdk.list_workspaces()
            >>> for ws in workspaces:
            ...     print(f"{ws.workspace_id}: {ws.status}")
        """
        logger.debug("Listing workspaces")
        data = await self._request("GET", "/v1/workspaces")
        return [Workspace(**item) for item in data]

    # =========================================================================
    # Dataset Management
    # =========================================================================

    async def list_datasets(self) -> List[DatasetInfo]:
        """
        List all available datasets.

        Returns:
            List of DatasetInfo objects

        Example:
            >>> datasets = await sdk.list_datasets()
            >>> for ds in datasets:
            ...     print(f"{ds.name}: {ds.size_mb} MB")
        """
        logger.debug("Listing datasets")
        data = await self._request("GET", "/v1/datasets")
        return [DatasetInfo(**item) for item in data]

    async def prepare_datasets(
        self,
        workspace_id: str,
        datasets: List[str],
        strategy: str = "copy",
    ) -> Dict[str, Any]:
        """
        Prepare datasets in workspace.

        Copies or links datasets to the workspace/data/ directory.

        Args:
            workspace_id: Workspace identifier
            datasets: List of dataset names to prepare
            strategy: Preparation strategy - "copy" or "link" (default: "copy")

        Returns:
            Response with status and prepared datasets

        Example:
            >>> result = await sdk.prepare_datasets(
            ...     workspace_id="my-experiment",
            ...     datasets=["titanic", "bike-sharing"]
            ... )
            >>> print(result["status"])
            prepared
        """
        logger.info(f"Preparing datasets {datasets} in workspace: {workspace_id}")
        payload = {
            "datasets": datasets,
            "strategy": strategy,
        }
        data = await self._request(
            "POST",
            f"/v1/workspaces/{workspace_id}/datasets",
            json=payload,
        )
        logger.info(f"Datasets prepared in workspace: {workspace_id}")
        return data

    async def list_workspace_datasets(
        self,
        workspace_id: str,
    ) -> List[DatasetInfo]:
        """
        List datasets prepared in a workspace.

        Args:
            workspace_id: Workspace identifier

        Returns:
            List of DatasetInfo objects prepared in the workspace

        Example:
            >>> datasets = await sdk.list_workspace_datasets("my-experiment")
        """
        logger.debug(f"Listing datasets in workspace: {workspace_id}")
        data = await self._request(
            "GET",
            f"/v1/workspaces/{workspace_id}/datasets",
        )
        return [DatasetInfo(**item) for item in data]

    # =========================================================================
    # Code Execution (Core Functionality)
    # =========================================================================

    async def execute(
        self,
        workspace_id: str,
        code: str,
        mode: Literal["safe", "fast", "secure"] = "safe",
        timeout_sec: int = 3600,
        datasets: Optional[List[str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        wait: bool = True,
    ) -> ExecutionResult:
        """
        Execute Python code in a workspace.

        This is the main method for running code in the sandbox.

        Args:
            workspace_id: Workspace identifier
            code: Python code to execute
            mode: Execution mode - "safe", "fast", or "secure" (default: "safe")
            timeout_sec: Timeout in seconds (default: 3600)
            datasets: List of datasets to prepare before execution
            env_vars: Environment variables to set in the sandbox
            wait: Whether to wait for execution to complete (default: True)

        Returns:
            ExecutionResult object with output and metadata

        Example:
            >>> result = await sdk.execute(
            ...     workspace_id="my-experiment",
            ...     code="print('Hello, World!')",
            ...     mode="fast"
            ... )
            >>> print(result.stdout)
            Hello, World!
        """
        payload = {
            "code": code,
            "mode": mode,
            "timeout_sec": timeout_sec,
            "datasets": datasets or [],
            "env_vars": env_vars or {},
        }

        logger.info(f"Executing code in workspace: {workspace_id}, mode: {mode}")

        # Start execution
        response = await self._request(
            "POST",
            f"/v1/workspaces/{workspace_id}/run",
            json=payload,
        )

        execution_id = response["execution_id"]
        logger.info(f"Execution started: {execution_id}")

        if wait:
            result = await self._wait_for_completion(execution_id, workspace_id, timeout_sec)
            return result
        else:
            # Return a partial result with execution_id
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="",
                isolation_level=mode,
                duration_ms=0,
            )

    async def _wait_for_completion(
        self,
        execution_id: str,
        workspace_id: str,
        timeout_sec: int,
    ) -> ExecutionResult:
        """
        Wait for execution to complete.

        Args:
            execution_id: Execution identifier
            workspace_id: Workspace identifier
            timeout_sec: Maximum time to wait

        Returns:
            ExecutionResult

        Raises:
            ExecutionTimeoutError: If execution times out
        """
        start_time = time.time()
        poll_interval = self.poll_interval

        while True:
            elapsed = time.time() - start_time

            if elapsed > timeout_sec:
                raise ExecutionTimeoutError(
                    execution_id=execution_id,
                    timeout_sec=timeout_sec,
                )

            # Check status
            status = await self.get_execution_status(workspace_id, execution_id)

            if status.status == "completed":
                # Fetch final result
                return await self._get_execution_result(execution_id, workspace_id)

            elif status.status == "failed":
                # Fetch error details
                result = await self._get_execution_result(execution_id, workspace_id)
                raise ExecutionFailedError(
                    execution_id=execution_id,
                    reason=result.stderr or "Unknown error",
                )

            elif status.status == "stopped":
                raise ExecutionFailedError(
                    execution_id=execution_id,
                    reason="Execution was stopped",
                )

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    async def _get_execution_result(
        self,
        execution_id: str,
        workspace_id: str,
    ) -> ExecutionResult:
        """
        Get execution result from logs or API.

        Args:
            execution_id: Execution identifier
            workspace_id: Workspace identifier

        Returns:
            ExecutionResult object
        """
        logs = await self.get_execution_logs(workspace_id, execution_id)
        status = await self.get_execution_status(workspace_id, execution_id)

        # Determine success from status
        success = status.status == "completed"

        return ExecutionResult(
            success=success,
            stdout=logs.logs,
            stderr="",
            execution_id=execution_id,
            workspace_id=workspace_id,
            backend=status.backend or "docker",
            isolation_level="fast",
            duration_ms=status.duration_ms if hasattr(status, 'duration_ms') and status.duration_ms else 0,
        )

    async def get_execution_status(
        self,
        workspace_id: str,
        execution_id: str,
    ) -> ExecutionStatus:
        """
        Get execution status.

        Args:
            workspace_id: Workspace identifier
            execution_id: Execution identifier

        Returns:
            ExecutionStatus object with current status

        Example:
            >>> status = await sdk.get_execution_status("my-experiment", "exec-123")
            >>> print(status.status)
            running
        """
        logger.debug(f"Getting execution status: {execution_id}")
        data = await self._request(
            "GET",
            f"/v1/workspaces/{workspace_id}/runs/{execution_id}",
        )
        return ExecutionStatus(**data)

    async def stop_execution(
        self,
        workspace_id: str,
        execution_id: str,
    ) -> Dict[str, str]:
        """
        Stop a running execution.

        Args:
            workspace_id: Workspace identifier
            execution_id: Execution identifier

        Returns:
            Response with status

        Example:
            >>> result = await sdk.stop_execution("my-experiment", "exec-123")
            >>> print(result["status"])
            stopped
        """
        logger.info(f"Stopping execution: {execution_id}")
        data = await self._request(
            "POST",
            f"/v1/workspaces/{workspace_id}/runs/{execution_id}/stop",
        )
        logger.info(f"Execution stopped: {execution_id}")
        return data

    async def get_execution_logs(
        self,
        workspace_id: str,
        execution_id: str,
        offset: int = 0,
        limit: int = 1000,
    ) -> "ExecutionLogs":
        """
        Get execution logs.

        Args:
            workspace_id: Workspace identifier
            execution_id: Execution identifier
            offset: Log offset (default: 0)
            limit: Maximum number of characters (default: 1000)

        Returns:
            ExecutionLogs object with log content

        Example:
            >>> logs = await sdk.get_execution_logs("my-experiment", "exec-123")
            >>> print(logs.logs)
            Hello, World!
        """
        logger.debug(f"Getting logs for execution: {execution_id}")
        params = {"offset": offset, "limit": limit}
        data = await self._request(
            "GET",
            f"/v1/workspaces/{workspace_id}/runs/{execution_id}/logs",
            params=params,
        )
        return ExecutionLogs(**data)

    # =========================================================================
    # Synchronous Wrapper Methods
    # =========================================================================

    def execute_sync(
        self,
        workspace_id: str,
        code: str,
        mode: Literal["safe", "fast", "secure"] = "safe",
        timeout_sec: int = 3600,
        datasets: Optional[List[str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Synchronous wrapper for execute.

        Note: This creates a new event loop. For better performance,
        use the async execute() method with asyncio.run().

        Args:
            Same as execute()

        Returns:
            ExecutionResult object
        """
        return asyncio.run(
            self.execute(
                workspace_id=workspace_id,
                code=code,
                mode=mode,
                timeout_sec=timeout_sec,
                datasets=datasets,
                env_vars=env_vars,
            )
        )

    def create_workspace_sync(
        self,
        workspace_id: str,
        setup_dirs: Optional[List[str]] = None,
    ) -> Workspace:
        """
        Synchronous wrapper for create_workspace.

        Args:
            Same as create_workspace()

        Returns:
            Workspace object
        """
        return asyncio.run(self.create_workspace(workspace_id, setup_dirs))

    def get_workspace_sync(self, workspace_id: str) -> Workspace:
        """
        Synchronous wrapper for get_workspace.

        Args:
            Same as get_workspace()

        Returns:
            Workspace object
        """
        return asyncio.run(self.get_workspace(workspace_id))

    def list_workspaces_sync(self) -> List[Workspace]:
        """
        Synchronous wrapper for list_workspaces.

        Returns:
            List of Workspace objects
        """
        return asyncio.run(self.list_workspaces())

    def delete_workspace_sync(self, workspace_id: str) -> None:
        """
        Synchronous wrapper for delete_workspace.

        Args:
            Same as delete_workspace()
        """
        asyncio.run(self.delete_workspace(workspace_id))

    def list_datasets_sync(self) -> List[DatasetInfo]:
        """
        Synchronous wrapper for list_datasets.

        Returns:
            List of DatasetInfo objects
        """
        return asyncio.run(self.list_datasets())

    def prepare_datasets_sync(
        self,
        workspace_id: str,
        datasets: List[str],
        strategy: str = "copy",
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for prepare_datasets.

        Args:
            Same as prepare_datasets()

        Returns:
            Response dict
        """
        return asyncio.run(
            self.prepare_datasets(workspace_id, datasets, strategy)
        )

    def get_execution_status_sync(
        self,
        workspace_id: str,
        execution_id: str,
    ) -> ExecutionStatus:
        """
        Synchronous wrapper for get_execution_status.

        Args:
            Same as get_execution_status()

        Returns:
            ExecutionStatus object
        """
        return asyncio.run(
            self.get_execution_status(workspace_id, execution_id)
        )

    def stop_execution_sync(
        self,
        workspace_id: str,
        execution_id: str,
    ) -> Dict[str, str]:
        """
        Synchronous wrapper for stop_execution.

        Args:
            Same as stop_execution()

        Returns:
            Response dict
        """
        return asyncio.run(self.stop_execution(workspace_id, execution_id))

    def get_execution_logs_sync(
        self,
        workspace_id: str,
        execution_id: str,
        offset: int = 0,
        limit: int = 1000,
    ) -> "ExecutionLogs":
        """
        Synchronous wrapper for get_execution_logs.

        Args:
            Same as get_execution_logs()

        Returns:
            ExecutionLogs object
        """
        return asyncio.run(
            self.get_execution_logs(workspace_id, execution_id, offset, limit)
        )


class ExecutionLogs(BaseModel):
    """Execution logs response"""
    execution_id: str = Field(..., description="Execution ID")
    logs: str = Field(..., description="Log content")
    offset: int = Field(..., description="Log offset")
    limit: int = Field(..., description="Log limit")
