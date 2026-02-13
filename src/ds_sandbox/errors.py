"""
ds-sandbox error definitions

Standard exceptions used across the ds-sandbox project.
"""

from typing import Optional, Dict, Any


class SandboxError(Exception):
    """Base exception for all sandbox errors"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN"
        self.details = details or {}

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class WorkspaceNotFoundError(SandboxError):
    """Workspace does not exist"""

    def __init__(self, workspace_id: str):
        super().__init__(
            message=f"Workspace '{workspace_id}' not found",
            error_code="WSP_NOT_FOUND"
        )
        self.workspace_id = workspace_id


class DatasetNotFoundError(SandboxError):
    """Dataset does not exist"""

    def __init__(self, dataset_name: str):
        super().__init__(
            message=f"Dataset '{dataset_name}' not found in registry",
            error_code="DAT_NOT_FOUND"
        )
        self.dataset_name = dataset_name


class DatasetNotPreparedError(SandboxError):
    """Dataset not prepared in workspace"""

    def __init__(self, dataset_name: str, workspace_id: str):
        super().__init__(
            message=f"Dataset '{dataset_name}' not prepared in workspace '{workspace_id}'",
            error_code="DAT_NOT_PREPARED"
        )
        self.dataset_name = dataset_name
        self.workspace_id = workspace_id


class ExecutionTimeoutError(SandboxError):
    """Execution timeout"""

    def __init__(self, execution_id: str, timeout_sec: int):
        super().__init__(
            message=f"Execution '{execution_id}' timed out after {timeout_sec}s",
            error_code="EXEC_TIMEOUT"
        )
        self.execution_id = execution_id
        self.timeout_sec = timeout_sec


class ExecutionFailedError(SandboxError):
    """Execution failed"""

    def __init__(self, execution_id: str, reason: str):
        super().__init__(
            message=f"Execution '{execution_id}' failed: {reason}",
            error_code="EXEC_FAILED"
        )
        self.execution_id = execution_id
        self.reason = reason


class ExecutionNotFoundError(SandboxError):
    """Execution not found"""

    def __init__(self, execution_id: str):
        super().__init__(
            message=f"Execution '{execution_id}' not found",
            error_code="EXEC_NOT_FOUND"
        )
        self.execution_id = execution_id


class ResourceLimitError(SandboxError):
    """Resource limit exceeded"""

    def __init__(
        self,
        resource_type: str,
        limit: float,
        actual: float
    ):
        super().__init__(
            message=f"{resource_type} limit exceeded: {actual}/{limit}",
            error_code="RES_LIMIT"
        )
        self.resource_type = resource_type
        self.limit = limit
        self.actual = actual


class BackendUnavailableError(SandboxError):
    """Backend not available"""

    def __init__(self, backend: str):
        super().__init__(
            message=f"Backend '{backend}' is not available",
            error_code="BACKEND_UNAVAILABLE"
        )
        self.backend = backend


class SecurityScanFailedError(SandboxError):
    """Security scan failed"""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Security scan failed: {reason}",
            error_code="SEC_SCAN_FAILED"
        )
        self.reason = reason


class InvalidRequestError(SandboxError):
    """Invalid request parameters"""

    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            message=f"Invalid field '{field}': {reason}",
            error_code="INVALID_REQUEST"
        )
        self.field = field
        self.value = value
        self.reason = reason


# Error codes
ERROR_CODES = {
    # Workspace errors (WSP_xxx)
    "WSP_NOT_FOUND": "Workspace does not exist",
    "WSP_INVALID": "Invalid workspace state",
    "DAT_NOT_FOUND": "Dataset not found",
    "DAT_NOT_PREPARED": "Dataset not prepared",

    # Execution errors (EXEC_xxx)
    "EXEC_TIMEOUT": "Execution timeout",
    "EXEC_FAILED": "Execution failed",

    # Resource errors (RES_xxx)
    "RES_LIMIT": "Resource limit exceeded",

    # Backend errors (BAK_xxx)
    "BACKEND_UNAVAILABLE": "Backend not available",

    # Security errors (SEC_xxx)
    "SEC_SCAN_FAILED": "Security scan failed",

    # Request errors (REQ_xxx)
    "INVALID_REQUEST": "Invalid request parameter",
}
