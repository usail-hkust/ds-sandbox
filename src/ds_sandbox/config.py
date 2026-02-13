from pydantic import BaseModel, Field


class SandboxConfig(BaseModel):
    """
    Runtime configuration for ds-sandbox.

    This configuration is loaded from:
    1. Environment variables (SANDBOX_*)
    2. Configuration file (if provided)
    3. Default values (hardcoded)

    Priority: Environment variables > Config file > Defaults
    """

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
    default_network_policy: str = Field(
        default="disabled",
        description="Default network access policy (disabled/whitelist/proxy)"
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
        - SANDBOX_NETWORK_POLICY: Network policy (disabled/whitelist/proxy)
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
