from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from ds_sandbox.types import StorageConfig


class SandboxConfig(BaseModel):
    """
    Runtime configuration for ds-sandbox.

    This configuration is loaded from:
    1. Environment variables (SANDBOX_*)
    2. Configuration file (if provided)
    3. Default values (hardcoded)

    Priority: Environment variables > Config file > Defaults
    """

    # API configuration (for remote usage)
    api_endpoint: str = Field(
        default="",
        description="Remote API endpoint (e.g., http://localhost:8000). If empty, use local backend."
    )

    # Backend selection
    default_backend: str = Field(
        default="docker",
        description="Default backend to use (docker/local/firecracker/kata/auto)"
    )

    default_isolation: str = Field(
        default="auto",
        description="Default isolation level (auto/fast/secure)"
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
        description="Days before unused workspaces are cleaned up"
    )

    # Paused workspace storage
    paused_workspaces_base_dir: str = Field(
        default="/opt/paused",
        description="Base directory for storing paused workspace states"
    )

    paused_workspace_retention_days: int = Field(
        default=30,
        ge=1,
        le=90,
        description="Days before paused workspaces are automatically cleaned up"
    )

    # Dataset management
    dataset_registry_dir: str = Field(
        default="/opt/datasets",
        description="Central dataset registry directory"
    )

    default_dataset_strategy: str = Field(
        default="copy",
        description="Default dataset preparation strategy (copy/link)"
    )

    # Security defaults
    allow_internet: bool = Field(
        default=True,
        description="Whether to allow internet access by default"
    )

    default_network_policy: str = Field(
        default="allow",
        description="Default network access policy (allow/deny/whitelist)"
    )

    network_whitelist: List[str] = Field(
        default_factory=list,
        description="Default network whitelist (domains/IPs allowed when network_policy=whitelist)"
    )

    default_timeout_sec: int = Field(
        default=3600,
        ge=1,
        le=86400,
        description="Default execution timeout (seconds)"
    )

    default_memory_mb: int = Field(
        default=4096,
        ge=512,
        le=65536,
        description="Default memory limit per execution (MB)"
    )

    enable_gpu_by_default: bool = Field(
        default=False,
        description="Enable GPU by default (requires secure backend)"
    )

    # Firecracker specific
    firecracker_pool_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of pre-warmed Firecracker VMs"
    )

    firecracker_memory_mb: int = Field(
        default=2048,
        ge=512,
        le=65536,
        description="Memory per Firecracker VM (MB)"
    )

    # API server
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )

    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="API server port"
    )

    # Monitoring
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics collection"
    )

    enable_audit_log: bool = Field(
        default=True,
        description="Enable audit logging"
    )

    audit_log_path: str = Field(
        default="/var/log/ds-sandbox/audit.logl",
        description="Path to audit log file"
    )

    # Default user and working directory
    default_user: str = Field(
        default="user",
        description="Default user for sandbox sessions"
    )

    default_workdir: str = Field(
        default="/home/user",
        description="Default working directory for sandbox sessions"
    )

    # Template configuration
    templates_dir: str = Field(
        default="/etc/ds-sandbox/templates",
        description="Directory containing template configurations"
    )

    # Storage configuration
    storage_mounts: Dict[str, StorageConfig] = Field(
        default_factory=dict,
        description="Storage bucket configurations (mount_name -> StorageConfig)"
    )

    # Default storage mount base directory
    storage_base_dir: str = Field(
        default="/workspace/storage",
        description="Base directory for mounting storage buckets in workspaces"
    )

    @classmethod
    def load_template(cls, template_id: str, templates_dir: Optional[str] = None) -> Optional["Template"]:
        """
        Load a template from the templates directory.

        Args:
            template_id: Template identifier (filename without extension)
            templates_dir: Optional custom templates directory

        Returns:
            Template object if found, None otherwise
        """
        from ds_sandbox.types import Template
        import yaml
        import os

        dir_path = templates_dir or cls().templates_dir
        template_path = os.path.join(dir_path, f"{template_id}.yaml")

        if not os.path.exists(template_path):
            template_path = os.path.join(dir_path, f"{template_id}.yml")

        if not os.path.exists(template_path):
            return None

        with open(template_path, "r") as f:
            data = yaml.safe_load(f)

        return Template(**data)

    @classmethod
    def list_templates(cls, templates_dir: Optional[str] = None) -> List[str]:
        """
        List available template IDs in the templates directory.

        Args:
            templates_dir: Optional custom templates directory

        Returns:
            List of template IDs
        """
        import os
        import glob

        dir_path = templates_dir or cls().templates_dir

        if not os.path.exists(dir_path):
            return []

        # Find all yaml/yml files
        patterns = [
            os.path.join(dir_path, "*.yaml"),
            os.path.join(dir_path, "*.yml"),
        ]

        template_ids = []
        for pattern in patterns:
            for filepath in glob.glob(pattern):
                filename = os.path.basename(filepath)
                # Remove extension
                template_id = os.path.splitext(filename)[0]
                template_ids.append(template_id)

        return sorted(set(template_ids))

    @classmethod
    def from_env(cls) -> "SandboxConfig":
        """
        Load configuration from environment variables.

        Environment variables (SANDBOX_*) override defaults:

        - SANDBOX_DEFAULT_BACKEND: Backend to use (docker/local/firecracker/kata/auto)
        - SANDBOX_DEFAULT_ISOLATION: Default isolation level
        - SANDBOX_WORKSPACE_BASE: Workspace base directory
        - SANDBOX_DATASET_DIR: Dataset registry directory
        - SANDBOX_TIMEOUT: Default timeout in seconds
        - SANDBOX_MEMORY_MB: Default memory limit
        - SANDBOX_NETWORK_POLICY: Network policy (allow/deny/whitelist)
        - SANDBOX_ALLOW_INTERNET: Allow internet access (true/false)
        - SANDBOX_NETWORK_WHITELIST: Comma-separated list of allowed domains/IPs
        - SANDBOX_ENABLE_GPU: Enable GPU (false)
        """
        import os

        kwargs = {}

        # Backend selection
        if "SANDBOX_DEFAULT_BACKEND" in os.environ:
            kwargs["default_backend"] = os.environ["SANDBOX_DEFAULT_BACKEND"]
        if "SANDBOX_DEFAULT_ISOLATION" in os.environ:
            kwargs["default_isolation"] = os.environ["SANDBOX_DEFAULT_ISOLATION"]

        # Workspace
        if "SANDBOX_WORKSPACE_BASE" in os.environ:
            kwargs["workspace_base_dir"] = os.environ["SANDBOX_WORKSPACE_BASE"]

        # Datasets
        if "SANDBOX_DATASET_DIR" in os.environ:
            kwargs["dataset_registry_dir"] = os.environ["SANDBOX_DATASET_DIR"]

        # Security
        if "SANDBOX_TIMEOUT" in os.environ:
            kwargs["default_timeout_sec"] = int(os.environ["SANDBOX_TIMEOUT"])
        if "SANDBOX_MEMORY_MB" in os.environ:
            kwargs["default_memory_mb"] = int(os.environ["SANDBOX_MEMORY_MB"])
        if "SANDBOX_NETWORK_POLICY" in os.environ:
            kwargs["default_network_policy"] = os.environ["SANDBOX_NETWORK_POLICY"]
        if "SANDBOX_ALLOW_INTERNET" in os.environ:
            kwargs["allow_internet"] = os.environ["SANDBOX_ALLOW_INTERNET"].lower() == "true"
        if "SANDBOX_NETWORK_WHITELIST" in os.environ:
            whitelist = os.environ["SANDBOX_NETWORK_WHITELIST"]
            kwargs["network_whitelist"] = [item.strip() for item in whitelist.split(",") if item.strip()]
        if "SANDBOX_ENABLE_GPU" in os.environ:
            kwargs["enable_gpu_by_default"] = os.environ["SANDBOX_ENABLE_GPU"].lower() == "true"

        return cls(**kwargs)

    @classmethod
    def from_file(cls, config_path: str) -> "SandboxConfig":
        """
        Load configuration from a YAML or JSON file.

        Supported formats: .yaml, .yml, .json
        """
        import yaml

        with open(config_path, "r") as f:
            if config_path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f)
            elif config_path.endswith(".json"):
                import json
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path}")

        return cls(**data)
