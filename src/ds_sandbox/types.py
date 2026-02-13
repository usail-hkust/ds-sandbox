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
    "SandboxInfo",
    "Template",
    "SandboxEvent",
    "PausedWorkspace",
    "SandboxMetrics",
    "StorageConfig",
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
    allow_internet: bool = Field(
        default=True,
        description="Whether to allow internet access"
    )

    network_policy: Literal["allow", "deny", "whitelist"] = Field(
        default="allow",
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
    allow_internet: bool = Field(
        default=True,
        description="Whether to allow internet access"
    )

    network_policy: Literal["allow", "deny", "whitelist"] = Field(
        default="allow",
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

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata for the sandbox"
    )


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


class SandboxInfo(BaseModel):
    """
    Sandbox information (E2B-compatible).

    This is returned by Sandbox.get_info() to provide details about the sandbox.
    """

    sandbox_id: str = Field(..., description="Unique sandbox identifier")

    template_id: Optional[str] = Field(None, description="Template ID used to create the sandbox")

    name: Optional[str] = Field(None, description="Sandbox name")

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    started_at: str = Field(..., description="Sandbox start timestamp (ISO 8601)")

    end_at: Optional[str] = Field(None, description="Sandbox end timestamp (ISO 8601)")


class Template(BaseModel):
    """
    E2B-compatible Template for custom sandbox configurations.

    Templates allow defining custom sandboxes with specific configurations
    including base image, environment variables, files, and startup commands.

    Usage:
        >>> from ds_sandbox import Template, Sandbox
        >>>
        >>> # Create a template
        >>> template = Template(
        ...     id="my-custom-template",
        ...     name="My Custom Template",
        ...     env={"CUSTOM_VAR": "value"},
        ...     files={"setup.sh": "#!/bin/bash\\necho 'Hello'"},
        ...     cmd=["bash", "/home/user/setup.sh"],
        ... )
        >>>
        >>> # Use the template when creating a sandbox
        >>> sandbox = Sandbox.create(template=template)
    """

    id: str = Field(..., description="Template identifier")

    name: Optional[str] = Field(None, description="Template name")

    description: Optional[str] = Field(None, description="Template description")

    # Base image (for Docker-based backends)
    image: Optional[str] = Field(None, description="Docker image to use")

    # Environment variables
    env: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to set in the sandbox"
    )

    # Files to copy into the sandbox
    files: Dict[str, str] = Field(
        default_factory=dict,
        description="Files to create in the sandbox (path -> content)"
    )

    # Commands to run on startup
    cmd: List[str] = Field(
        default_factory=list,
        description="Commands to run on startup"
    )

    # Start command (overrides default)
    start_cmd: Optional[str] = Field(
        None,
        description="Command to start the sandbox (for persistent backends)"
    )

    # Ready command to check if sandbox is ready
    ready_cmd: Optional[str] = Field(
        None,
        description="Command to check if sandbox is ready"
    )

    # Ready command timeout in seconds
    ready_timeout: int = Field(
        default=20,
        ge=1,
        le=300,
        description="Timeout for ready command in seconds"
    )

    # Default user
    user: Optional[str] = Field(
        None,
        description="Default user for the sandbox"
    )

    # Default working directory
    workdir: Optional[str] = Field(
        None,
        description="Default working directory"
    )

    # CPU count
    cpu_count: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Number of CPU cores"
    )

    # Memory in MB
    memory_mb: int = Field(
        default=2048,
        ge=256,
        le=65536,
        description="Memory in MB"
    )

    # Files to copy (as list of dicts with src/dest)
    copy_files: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Files to copy into template (list of {src, dest} dicts)"
    )

    # Commands to run during build
    run_cmds: List[str] = Field(
        default_factory=list,
        description="Commands to run during build"
    )

    # Template aliases for easy referencing
    aliases: List[str] = Field(
        default_factory=list,
        description="Template aliases for easy referencing"
    )

    # Skip cache during build
    skip_cache: bool = Field(
        default=False,
        description="Skip cache during build"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional template metadata"
    )


class SandboxEvent(BaseModel):
    """
    E2B-compatible Sandbox Event model.

    Represents lifecycle events for sandboxes (workspaces) including
    creation, updates, and deletion events.

    Example:
        >>> event = SandboxEvent(
        ...     id="evt-123456",
        ...     type="sandbox.lifecycle.created",
        ...     event_data={"timeout": 3600},
        ...     sandbox_id="sandbox-abc",
        ...     workspace_id="my-workspace",
        ...     timestamp="2024-01-01T12:00:00Z"
        ... )
    """

    id: str = Field(..., description="Event ID (UUID)")

    type: str = Field(
        ...,
        description="Event type (e.g., sandbox.lifecycle.created, sandbox.lifecycle.updated, sandbox.lifecycle.killed)"
    )

    event_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event data (can contain set_timeout, etc.)"
    )

    sandbox_id: str = Field(..., description="Sandbox ID")

    workspace_id: str = Field(..., description="Workspace ID")

    timestamp: str = Field(..., description="ISO 8601 timestamp")


class PausedWorkspace(BaseModel):
    """
    Paused workspace metadata.

    Represents a paused sandbox workspace with its saved state information.
    Paused workspaces can be resumed to restore the filesystem and runtime state.

    Example:
        >>> paused = PausedWorkspace(
        ...     workspace_id="my-workspace",
        ...     backup_path="/opt/paused/my-workspace-20240101",
        ...     original_path="/opt/workspaces/my-workspace",
        ...     paused_at="2024-01-01T12:00:00Z",
        ...     expires_at="2024-01-31T12:00:00Z",
        ...     files_count=42,
        ...     size_mb=128.5
        ... )
    """

    workspace_id: str = Field(..., description="Original workspace ID")

    backup_path: str = Field(..., description="Path where workspace state is stored")

    original_path: str = Field(..., description="Original workspace path before pause")

    paused_at: str = Field(..., description="ISO 8601 timestamp when workspace was paused")

    expires_at: str = Field(..., description="ISO 8601 timestamp when paused state expires (30 days default)")

    files_count: int = Field(default=0, description="Number of files in the paused workspace")

    size_mb: float = Field(default=0.0, description="Size of paused workspace in MB")

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the paused workspace"
    )


class SandboxMetrics(BaseModel):
    """
    E2B-compatible Sandbox Metrics model.

    Represents CPU and memory metrics for a sandbox workspace at a specific point in time.

    Example:
        >>> metrics = SandboxMetrics(
        ...     cpu_count=4,
        ...     cpu_used_pct=25.5,
        ...     mem_total_mib=8192,
        ...     mem_used_mib=2048,
        ...     timestamp="2024-01-01T12:00:00Z"
        ... )
    """

    cpu_count: int = Field(..., description="Number of CPU cores")

    cpu_used_pct: float = Field(..., description="CPU usage percentage (0-100)")

    mem_total_mib: int = Field(..., description="Total memory in MiB")

    mem_used_mib: int = Field(..., description="Used memory in MiB")

    timestamp: str = Field(..., description="ISO 8601 timestamp of the metrics snapshot")


class StorageConfig(BaseModel):
    """
    Storage bucket configuration.

    This model defines the configuration for mounting cloud storage (S3, GCS, Azure)
    to a sandbox workspace.

    Example:
        >>> config = StorageConfig(
        ...     provider="s3",
        ...     bucket="my-data-bucket",
        ...     region="us-east-1",
        ...     path_prefix="data/",
        ...     credentials={"access_key_id": "AKIAIOSFODNN7EXAMPLE", "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"}
        ... )
    """

    provider: Literal["s3", "gcs", "azure", "local"] = Field(
        ...,
        description="Storage provider: s3 (AWS S3), gcs (Google Cloud Storage), azure (Azure Blob Storage), local (local filesystem)"
    )

    bucket: str = Field(..., description="Bucket name or container name")

    region: Optional[str] = Field(
        None,
        description="Region for S3 buckets (e.g., us-east-1, eu-west-1)"
    )

    path_prefix: Optional[str] = Field(
        None,
        description="Path prefix within the bucket to mount"
    )

    credentials: Optional[Dict[str, str]] = Field(
        None,
        description="Credentials for accessing the storage. For S3: access_key_id, secret_access_key. For GCS: service_account_key. For Azure: account_name, account_key"
    )

    # Mount options
    read_only: bool = Field(
        default=False,
        description="Whether to mount storage as read-only"
    )

    mount_point: Optional[str] = Field(
        None,
        description="Target mount point within the workspace (default: /workspace/storage/<bucket_name>)"
    )

