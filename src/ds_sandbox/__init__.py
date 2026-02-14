"""
ds-sandbox: Workspace-first AI code execution sandbox

General-purpose sandbox for AI agent code execution with
workspace-first data management and pluggable isolation backends.

This module provides an E2B-compatible API for ease of use.
"""

__version__ = "1.0.0"

from ds_sandbox.manager import SandboxManager
from ds_sandbox.config import SandboxConfig
from ds_sandbox.types import (
    ExecutionRequest,
    ExecutionResult,
    Workspace,
    DatasetInfo,
    CodeScanResult,
    SandboxInfo,
    Template,
    SandboxEvent,
    PausedWorkspace,
    SandboxMetrics,
)
from ds_sandbox.template import TemplateBuilder, BuildOptions
from ds_sandbox.api.rest import create_app, app, SandboxErrorResponse
from ds_sandbox.api.sdk import SandboxSDK, ExecutionStatus, ExecutionLogs

# E2B-compatible Sandbox class
from ds_sandbox.sandbox.sandbox import Sandbox

# E2B-compatible exception classes
from ds_sandbox.errors import (
    SandboxError,
    WorkspaceNotFoundError,
    DatasetNotFoundError,
    DatasetNotPreparedError,
    ExecutionTimeoutError,
    ExecutionFailedError,
    ExecutionNotFoundError,
    ResourceLimitError,
    BackendUnavailableError,
    SecurityScanFailedError,
    InvalidRequestError,
)

# Re-export core classes
__all__ = [
    "SandboxManager",
    "SandboxConfig",
    "ExecutionRequest",
    "ExecutionResult",
    "Workspace",
    "DatasetInfo",
    "CodeScanResult",
    "SandboxInfo",
    "Template",
    "SandboxEvent",
    "PausedWorkspace",
    "SandboxMetrics",
    # Template builder
    "TemplateBuilder",
    "BuildOptions",
    # API exports
    "create_app",
    "app",
    "SandboxErrorResponse",
    # SDK exports
    "SandboxSDK",
    "ExecutionStatus",
    "ExecutionLogs",
    # E2B-compatible API
    "Sandbox",
    # Exception classes
    "SandboxError",
    "WorkspaceNotFoundError",
    "DatasetNotFoundError",
    "DatasetNotPreparedError",
    "ExecutionTimeoutError",
    "ExecutionFailedError",
    "ExecutionNotFoundError",
    "ResourceLimitError",
    "BackendUnavailableError",
    "SecurityScanFailedError",
    "InvalidRequestError",
]
