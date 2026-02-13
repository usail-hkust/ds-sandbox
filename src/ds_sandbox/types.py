"""
ds-sandbox type definitions

Common types used across the ds-sandbox project.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

__all__ = [
    "ExecuteCodeRequest",
    "ExecutionRequest",
    "ExecutionResult",
    "Workspace",
    "DatasetInfo",
    "CodeScanResult",
    "SandboxConfig",
]


class ExecuteCodeRequest(BaseModel):
    """Request body for code execution (workspace_id from path)"""

    # Basic parameters
    code: str = Field(..., description="Python code to execute")

    # Data preparation
    datasets: List[str] = Field(
        default_factory=list,
        description="Dataset names to prepare in workspace/data/"
    )

    data_mounts: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom data mounts (path mapping)"
    )

    # Execution control
    mode: Literal["safe", "fast", "secure"] = Field(
        default="safe",
        description="Execution mode (affects backend selection)"
    )

    timeout_sec: int = Field(
        default=3600,
        ge=1,
        le=86400,
        description="Timeout in seconds"
    )

    # Resource limits
    memory_mb: int = Field(
        default=4096,
        ge=512,
        le=65536,
        description="Memory limit in MB"
    )

    cpu_cores: float = Field(
        default=2.0,
        ge=0.5,
        le=16.0,
        description="Number of CPU cores"
    )

    enable_gpu: bool = Field(
        default=False,
        description="Enable GPU access"
    )

    # Security configuration
    network_policy: Literal["disabled", "whitelist", "proxy"] = Field(
        default="disabled",
        description="Network access policy"
    )

    network_whitelist: List[str] = Field(
        default_factory=list,
        description="Network whitelist (when network_policy=whitelist)"
    )

    # Environment variables
    env_vars: Dict[str, str] = Field(
        default_factory=dict,
        description="Execution environment variables"
    )


class ExecutionRequest(BaseModel):
    """Code execution request (used internally with workspace_id)"""

    # Basic parameters
    code: str = Field(..., description="Python code to execute")

    workspace_id: str = Field(
        ...,
        description="Workspace ID",
        min_length=1,
        max_length=64
    )

    # Data preparation
    datasets: List[str] = Field(
        default_factory=list,
        description="Dataset names to prepare in workspace/data/"
    )

    data_mounts: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom data mounts (path mapping)"
    )

    # Execution control
    mode: Literal["safe", "fast", "secure"] = Field(
        default="safe",
        description="Execution mode (affects backend selection)"
    )

    timeout_sec: int = Field(
        default=3600,
        ge=1,
        le=86400,
        description="Timeout in seconds"
    )

    # Resource limits
    memory_mb: int = Field(
        default=4096,
        ge=512,
        le=65536,
        description="Memory limit in MB"
    )

    cpu_cores: float = Field(
        default=2.0,
        ge=0.5,
        le=16.0,
        description="Number of CPU cores"
    )

    enable_gpu: bool = Field(
        default=False,
        description="Enable GPU access"
    )

    # Security configuration
    network_policy: Literal["disabled", "whitelist", "proxy"] = Field(
        default="disabled",
        description="Network access policy"
    )

    network_whitelist: List[str] = Field(
        default_factory=list,
        description="Network whitelist (when network_policy=whitelist)"
    )

    # Environment variables
    env_vars: Dict[str, str] = Field(
        default_factory=dict,
        description="Execution environment variables"
    )


class ExecutionResult(BaseModel):
    """Execution result"""

    success: bool = Field(..., description="Execution succeeded")

    stdout: str = Field(..., description="Standard output")

    stderr: str = Field(default="", description="Standard error output")

    # Execution details
    exit_code: Optional[int] = Field(None, description="Exit code (if available)")

    duration_ms: int = Field(..., description="Execution duration in milliseconds")

    # Artifacts generated
    artifacts: List[str] = Field(
        default_factory=list,
        description="Generated file paths (relative to workspace)"
    )

    # Metadata
    execution_id: str = Field(..., description="Unique execution ID")

    workspace_id: str = Field(..., description="Workspace ID")

    backend: str = Field(..., description="Backend used")

    isolation_level: str = Field(..., description="Actual isolation level")

    # Audit information
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class Workspace(BaseModel):
    """Workspace information"""

    workspace_id: str = Field(
        ...,
        description="Workspace unique identifier"
    )

    host_path: str = Field(..., description="Host machine path")

    guest_path: str = Field(default="/workspace", description="Guest (sandbox) path")

    subdirs: List[str] = Field(
        default=["data", "models", "outputs"],
        description="Workspace subdirectories"
    )

    status: Literal["creating", "ready", "archived"] = Field(
        default="ready",
        description="Workspace status"
    )

    created_at: str = Field(..., description="Creation timestamp (ISO 8601)")

    last_used_at: Optional[str] = Field(None, description="Last used timestamp")


class DatasetInfo(BaseModel):
    """Dataset information"""

    name: str = Field(..., description="Dataset name")

    source_path: str = Field(..., description="Source path in central registry")

    size_mb: float = Field(..., ge=0, description="Size in MB")

    checksum: str = Field(..., description="SHA-256 checksum of the file")

    format: Literal["csv", "parquet", "json", "excel", "feather"] = Field(
        ...,
        description="Data format"
    )

    description: Optional[str] = Field(None, description="Dataset description")

    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization"
    )

    registered_at: str = Field(..., description="Registration timestamp (ISO 8601)")


class CodeScanResult(BaseModel):
    """Code scan result"""

    is_safe: bool = Field(..., description="Code is safe")

    risk_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Risk score (0.0 - 1.0)"
    )

    issues: List["CodeIssue"] = Field(
        default_factory=list,
        description="List of security issues found"
    )

    recommended_backend: str = Field(
        ...,
        description="Recommended backend name"
    )


class CodeIssue(BaseModel):
    """Code security issue"""

    type: str = Field(..., description="Issue type")

    line: int = Field(..., ge=1, description="Line number")

    severity: str = Field(..., description="Severity: low/medium/high")

    weight: float = Field(default=0.5, description="Risk weight")

    function: Optional[str] = Field(None, description="Related function")

    module: Optional[str] = Field(None, description="Related module")


class SandboxConfig(BaseModel):
    """Sandbox configuration"""

    # Backend selection
    default_backend: Literal["docker", "local", "firecracker", "kata", "auto"] = Field(
        default="auto",
        description="Default backend"
    )

    default_isolation: Literal["auto", "fast", "secure"] = Field(
        default="auto",
        description="Default isolation level"
    )

    # Jupyter execution (for chart support)
    use_jupyter: bool = Field(
        default=False,
        description="Use Jupyter kernel for code execution (enables chart capture like E2B)"
    )

    # Workspace management
    workspace_base_dir: str = Field(
        default="/opt/workspaces",
        description="Base directory for workspaces"
    )

    workspace_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Workspace retention before cleanup"
    )

    # Dataset management
    dataset_registry_dir: str = Field(
        default="/opt/datasets",
        description="Central dataset registry directory"
    )

    default_dataset_strategy: Literal["copy", "link"] = Field(
        default="copy",
        description="Default dataset preparation strategy"
    )

    # Security defaults
    default_network_policy: Literal["disabled", "whitelist"] = Field(
        default="disabled",
        description="Default network access policy"
    )

    default_timeout_sec: int = Field(
        default=3600,
        ge=1,
        le=86400,
        description="Default timeout (seconds)"
    )

    default_memory_mb: int = Field(
        default=4096,
        ge=512,
        le=65536,
        description="Default memory limit (MB)"
    )
