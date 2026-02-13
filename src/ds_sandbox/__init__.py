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
    SandboxConfig as SandboxConfigTypes,
)
from ds_sandbox.api.rest import create_app, app, SandboxErrorResponse
from ds_sandbox.api.sdk import SandboxSDK, ExecutionStatus, ExecutionLogs

# E2B-compatible Sandbox class
from ds_sandbox.sandbox import Sandbox

# Re-export core classes
__all__ = [
    "SandboxManager",
    "SandboxConfig",
    "ExecutionRequest",
    "ExecutionResult",
    "Workspace",
    "DatasetInfo",
    "CodeScanResult",
    "SandboxConfigTypes",
    "create_app",
    "app",
    "SandboxErrorResponse",
    # SDK exports
    "SandboxSDK",
    "ExecutionStatus",
    "ExecutionLogs",
    # E2B-compatible API
    "Sandbox",
]
