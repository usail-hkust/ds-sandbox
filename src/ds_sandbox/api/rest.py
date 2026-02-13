"""
FastAPI REST API Server for ds-sandbox

REST API, Python SDK, and MCP server.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ds_sandbox.config import SandboxConfig
from ds_sandbox.path_utils import validate_path_component
from ds_sandbox.workspace.service import WorkspaceService
from ds_sandbox.types import (
    ExecuteCodeRequest,
    ExecutionRequest,
    Workspace,
    DatasetInfo,
    SandboxEvent,
    PausedWorkspace,
    SandboxMetrics,
    Template,
    StorageConfig,
)
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

logger = logging.getLogger(__name__)

# =============================================================================
# Request/Response Models
# =============================================================================

class CreateWorkspaceRequest(BaseModel):
    """Request to create a new workspace"""
    workspace_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique workspace identifier"
    )
    setup_dirs: List[str] = Field(
        default_factory=lambda: ["data", "models", "outputs"],
        description="Additional subdirectories to create"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata for the sandbox"
    )


class PrepareDatasetsRequest(BaseModel):
    """Request to prepare datasets in workspace"""
    datasets: List[str] = Field(
        ...,
        min_length=1,
        description="List of dataset names to prepare"
    )
    strategy: str = Field(
        default="copy",
        description="Preparation strategy: copy or link"
    )


class MountStorageRequest(BaseModel):
    """Request to mount storage to a workspace"""
    storage_config: StorageConfig = Field(..., description="Storage configuration")
    mount_name: Optional[str] = Field(
        None,
        description="Optional name for this mount (defaults to bucket name)"
    )


class StorageMountResponse(BaseModel):
    """Storage mount response"""
    mount_name: str = Field(..., description="Mount name")
    provider: str = Field(..., description="Storage provider")
    bucket: str = Field(..., description="Bucket name")
    mount_point: str = Field(..., description="Mount point in workspace")
    path_prefix: Optional[str] = Field(None, description="Path prefix in bucket")
    read_only: bool = Field(..., description="Whether mount is read-only")


class StorageMountListResponse(BaseModel):
    """Storage mount list response"""
    mounts: List[StorageMountResponse] = Field(
        default_factory=list,
        description="List of storage mounts"
    )


class HealthStatus(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    backends: Dict[str, Any] = Field(
        default_factory=dict,
        description="Backend status information"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Response timestamp"
    )


class ExecutionInfo(BaseModel):
    """Execution start response"""
    execution_id: str = Field(..., description="Unique execution ID")
    workspace_id: str = Field(..., description="Workspace ID")
    status: str = Field(default="running", description="Execution status")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Creation timestamp"
    )


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


class ExecutionLogs(BaseModel):
    """Execution logs response"""
    execution_id: str = Field(..., description="Execution ID")
    logs: str = Field(..., description="Log content")
    offset: int = Field(..., description="Log offset")
    limit: int = Field(..., description="Log limit")


class SandboxErrorResponse(BaseModel):
    """统一错误响应"""
    error_code: str = Field(..., description="错误代码（SBX-XXX）")
    message: str = Field(..., description="用户友好的错误描述")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="额外错误详情"
    )
    request_id: str = Field(..., description="请求追踪ID")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="错误时间（ISO 8601）"
    )


class SystemMetrics(BaseModel):
    """系统指标响应"""
    total_executions: int = Field(default=0, description="Total executions")
    successful_executions: int = Field(default=0, description="Successful executions")
    failed_executions: int = Field(default=0, description="Failed executions")
    active_workspaces: int = Field(default=0, description="Active workspaces")
    avg_execution_time_ms: float = Field(default=0.0, description="Average execution time")


class WorkspaceMetricsResponse(BaseModel):
    """Workspace metrics response"""
    workspace_id: str = Field(..., description="Workspace ID")
    metrics: List[SandboxMetrics] = Field(default_factory=list, description="Metrics history")


# =============================================================================
# Template Request/Response Models
# =============================================================================

class BuildTemplateRequest(BaseModel):
    """Request to build a new template"""
    template: Template = Field(..., description="Template configuration")
    alias: Optional[str] = Field(None, description="Primary alias for the template")
    wait_timeout: int = Field(default=60, description="Wait timeout in seconds")
    debug: bool = Field(default=False, description="Enable debug mode")


class TemplateResponse(BaseModel):
    """Template response"""
    template: Template = Field(..., description="Template details")


class TemplateListResponse(BaseModel):
    """Template list response"""
    templates: List[Template] = Field(default_factory=list, description="List of templates")


# =============================================================================
# Error Code Mapping
# =============================================================================

ERROR_CODE_MAP = {
    WorkspaceNotFoundError: 404,
    DatasetNotFoundError: 404,
    DatasetNotPreparedError: 400,
    ExecutionTimeoutError: 408,
    ExecutionFailedError: 500,
    ExecutionNotFoundError: 404,
    ResourceLimitError: 413,
    BackendUnavailableError: 503,
    SecurityScanFailedError: 400,
    InvalidRequestError: 400,
}


# =============================================================================
# FastAPI Application
# =============================================================================

def create_app(config: Optional[SandboxConfig] = None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        config: Optional sandbox configuration

    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = SandboxConfig.from_env()

    app = FastAPI(
        title="ds-sandbox API",
        version="1.0.0",
        description="General-purpose AI code execution sandbox",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Store config in app state
    app.state.config = config

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register exception handlers
    register_exception_handlers(app)

    # Register routes
    register_routes(app)

    logger.info(f"FastAPI application created with config: {config}")
    return app


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers"""

    @app.exception_handler(SandboxError)
    async def sandbox_error_handler(request: Request, exc: SandboxError):
        """Handle SandboxError exceptions"""
        status_code = ERROR_CODE_MAP.get(type(exc), 500)
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        error_response = SandboxErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            request_id=request_id,
        )

        logger.error(
            f"SandboxError: {exc.error_code} - {exc.message}",
            extra={"request_id": request_id}
        )

        return JSONResponse(
            status_code=status_code,
            content=error_response.model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        logger.error(
            f"Unexpected error: {exc}",
            exc_info=True,
            extra={"request_id": request_id}
        )

        error_response = SandboxErrorResponse(
            error_code="INTERNAL_ERROR",
            message=str(exc) if exc else "An unexpected error occurred",
            details={},
            request_id=request_id,
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(),
        )


def register_routes(app: FastAPI) -> None:
    """Register all API routes"""

    # Dependency to get manager
    async def get_manager():
        """Get sandbox manager instance"""
        from ds_sandbox.manager import SandboxManager
        if not hasattr(app.state, "manager"):
            app.state.manager = SandboxManager(app.state.config)
        return app.state.manager

    # Request ID middleware
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    # =========================================================================
    # Health & Metrics
    # =========================================================================

    @app.get("/v1/health", response_model=HealthStatus, tags=["System"])
    async def health_check(manager=Depends(get_manager)):
        """
        System health check endpoint

        Returns overall health status and backend availability.
        """
        logger.info("Health check requested")

        try:
            backends = await manager.health_check()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            backends = {"error": str(e)}

        all_healthy = all(
            status.get("status") in ("healthy", "available")
            for status in backends.values()
        )

        return HealthStatus(
            status="healthy" if all_healthy else "degraded",
            version="1.0.0",
            backends=backends,
        )

    @app.get("/v1/metrics", response_model=SystemMetrics, tags=["System"])
    async def get_metrics(manager=Depends(get_manager)):
        """
        Get system metrics

        Returns current system metrics and statistics.
        """
        logger.info("Metrics requested")
        metrics = manager.get_metrics()
        return SystemMetrics(
            total_executions=metrics.get("total_executions", 0),
            successful_executions=metrics.get("successful_executions", 0),
            failed_executions=metrics.get("failed_executions", 0),
            active_workspaces=metrics.get("active_workspaces", 0),
            avg_execution_time_ms=metrics.get("avg_execution_time_ms", 0.0),
        )

    @app.get(
        "/v1/workspaces/{workspace_id}/metrics",
        response_model=WorkspaceMetricsResponse,
        tags=["Workspaces"]
    )
    async def get_workspace_metrics(
        workspace_id: str,
        manager=Depends(get_manager)
    ):
        """
        Get metrics for a workspace

        Returns metrics history for a specific workspace.
        """
        logger.info(f"Getting metrics for workspace: {workspace_id}")

        # Validate workspace exists
        await manager.get_workspace(workspace_id)

        # Collect current metrics
        current_metrics = manager.collect_workspace_metrics(workspace_id)

        # Get metrics history
        metrics_history = manager.get_workspace_metrics(workspace_id)

        # Ensure current metrics is in the history
        if not metrics_history or metrics_history[-1].timestamp != current_metrics.timestamp:
            metrics_history.append(current_metrics)

        return WorkspaceMetricsResponse(
            workspace_id=workspace_id,
            metrics=metrics_history,
        )

    @app.get(
        "/v1/metrics/system",
        response_model=List[SandboxMetrics],
        tags=["System"]
    )
    async def get_system_sandbox_metrics(
        manager=Depends(get_manager)
    ):
        """
        Get all system sandbox metrics

        Returns CPU and memory metrics across all workspaces.
        """
        logger.info("Getting all system sandbox metrics")
        return manager.get_system_metrics()

    # =========================================================================
    # Workspace Management
    # =========================================================================

    @app.post(
        "/v1/workspaces",
        response_model=Workspace,
        status_code=201,
        tags=["Workspaces"]
    )
    async def create_workspace(
        request: CreateWorkspaceRequest,
        manager=Depends(get_manager)
    ):
        """
        Create a new workspace

        Creates directory structure and initializes workspace metadata.
        """
        logger.info(f"Creating workspace: {request.workspace_id}")

        workspace = await manager.create_workspace(
            workspace_id=request.workspace_id,
            setup_dirs=request.setup_dirs,
            metadata=request.metadata,
        )

        logger.info(f"Workspace created: {workspace.workspace_id}")
        return workspace

    @app.get("/v1/workspaces", response_model=List[Workspace], tags=["Workspaces"])
    async def list_workspaces(
        state: Optional[str] = None,
        manager=Depends(get_manager)
    ):
        """
        List all workspaces

        Returns a list of all available workspaces.
        Can filter by state: "running" or "paused".
        """
        logger.info(f"Listing workspaces (state filter: {state})")
        return await manager.list_workspaces(state=state)

    @app.get("/v1/workspaces/{workspace_id}", response_model=Workspace, tags=["Workspaces"])
    async def get_workspace(
        workspace_id: str,
        manager=Depends(get_manager)
    ):
        """
        Get workspace information

        Returns detailed information about a specific workspace.
        """
        logger.info(f"Getting workspace: {workspace_id}")

        try:
            workspace = await manager.get_workspace(workspace_id)
            return workspace
        except WorkspaceNotFoundError:
            raise

    @app.delete(
        "/v1/workspaces/{workspace_id}",
        status_code=204,
        tags=["Workspaces"]
    )
    async def delete_workspace(
        workspace_id: str,
        manager=Depends(get_manager)
    ):
        """
        Delete a workspace

        Removes workspace and all associated data.
        """
        logger.info(f"Deleting workspace: {workspace_id}")

        await manager.delete_workspace(workspace_id)
        logger.info(f"Workspace deleted: {workspace_id}")

    # =========================================================================
    # Workspace Pause/Resume (E2B-compatible)
    # =========================================================================

    @app.post(
        "/v1/workspaces/{workspace_id}/pause",
        response_model=PausedWorkspace,
        tags=["Workspaces"]
    )
    async def pause_workspace(
        workspace_id: str,
        manager=Depends(get_manager)
    ):
        """
        Pause a workspace

        Saves the workspace state (filesystem) to a backup directory.
        The workspace can be resumed later to restore its state.
        Paused workspaces are stored for up to 30 days.
        """
        logger.info(f"Pausing workspace: {workspace_id}")

        paused_workspace = await manager.pause_workspace(workspace_id)
        logger.info(f"Workspace paused: {workspace_id}")

        return paused_workspace

    @app.post(
        "/v1/workspaces/{workspace_id}/resume",
        response_model=Workspace,
        tags=["Workspaces"]
    )
    async def resume_workspace(
        workspace_id: str,
        manager=Depends(get_manager)
    ):
        """
        Resume a paused workspace

        Restores the workspace from its saved state.
        """
        logger.info(f"Resuming workspace: {workspace_id}")

        workspace = await manager.resume_workspace(workspace_id)
        logger.info(f"Workspace resumed: {workspace_id}")

        return workspace

    @app.get(
        "/v1/workspaces/paused",
        response_model=List[PausedWorkspace],
        tags=["Workspaces"]
    )
    async def list_paused_workspaces(
        manager=Depends(get_manager)
    ):
        """
        List all paused workspaces

        Returns a list of all paused workspaces.
        """
        logger.info("Listing paused workspaces")
        return await manager.list_paused_workspaces()

    @app.get(
        "/v1/workspaces/{workspace_id}/pause",
        response_model=PausedWorkspace,
        tags=["Workspaces"]
    )
    async def get_paused_workspace(
        workspace_id: str,
        manager=Depends(get_manager)
    ):
        """
        Get paused workspace information

        Returns information about a paused workspace.
        """
        logger.info(f"Getting paused workspace: {workspace_id}")

        paused_workspace = await manager.get_paused_workspace(workspace_id)
        return paused_workspace

    @app.delete(
        "/v1/workspaces/{workspace_id}/pause",
        status_code=204,
        tags=["Workspaces"]
    )
    async def delete_paused_workspace(
        workspace_id: str,
        manager=Depends(get_manager)
    ):
        """
        Delete a paused workspace backup

        Removes the paused workspace backup without affecting any existing workspace.
        """
        logger.info(f"Deleting paused workspace: {workspace_id}")

        await manager.delete_paused_workspace(workspace_id)
        logger.info(f"Paused workspace deleted: {workspace_id}")

    # =========================================================================
    # Sandbox Lifecycle Events (E2B-compatible)
    # =========================================================================

    @app.get(
        "/v1/workspaces/{workspace_id}/events",
        response_model=List[SandboxEvent],
        tags=["Events"]
    )
    async def get_workspace_events(
        workspace_id: str,
        limit: int = 10,
        manager=Depends(get_manager)
    ):
        """
        Get events for a workspace

        Returns lifecycle events for a specific workspace.
        """
        logger.info(f"Getting events for workspace: {workspace_id}")
        return manager.get_events(workspace_id=workspace_id, limit=limit)

    @app.get(
        "/v1/workspaces/{workspace_id}/timeout",
        tags=["Workspaces"]
    )
    async def get_workspace_timeout(
        workspace_id: str,
        manager=Depends(get_manager)
    ):
        """
        Get workspace timeout

        Returns the current timeout setting for a workspace.
        """
        logger.info(f"Getting timeout for workspace: {workspace_id}")
        # Validate workspace exists
        await manager.get_workspace(workspace_id)
        # Return default timeout (can be extended to store per-workspace timeout)
        return {"workspace_id": workspace_id, "timeout_sec": 3600}

    @app.put(
        "/v1/workspaces/{workspace_id}/timeout",
        tags=["Workspaces"]
    )
    async def set_workspace_timeout(
        workspace_id: str,
        timeout_sec: int,
        manager=Depends(get_manager)
    ):
        """
        Set workspace timeout

        Updates the timeout for a workspace and emits an updated event.
        """
        logger.info(f"Setting timeout for workspace {workspace_id}: {timeout_sec}s")

        # Validate workspace exists
        await manager.get_workspace(workspace_id)

        # Emit lifecycle event
        manager.emit_event(
            event_type="sandbox.lifecycle.updated",
            workspace_id=workspace_id,
            event_data={"set_timeout": timeout_sec},
        )

        return {"workspace_id": workspace_id, "timeout_sec": timeout_sec}

    @app.get(
        "/v1/events",
        response_model=List[SandboxEvent],
        tags=["Events"]
    )
    async def get_all_events(
        limit: int = 100,
        manager=Depends(get_manager)
    ):
        """
        Get all events (admin)

        Returns all lifecycle events across all workspaces.
        """
        logger.info(f"Getting all events (limit={limit})")
        return manager.get_all_events(limit=limit)

    # =========================================================================
    # Dataset Management
    # =========================================================================

    @app.get("/v1/datasets", response_model=List[DatasetInfo], tags=["Datasets"])
    async def list_datasets(manager=Depends(get_manager)):
        """
        List available datasets

        Returns all registered datasets in the central registry.
        """
        from ds_sandbox.data.registry import DatasetRegistry

        logger.info("Listing datasets")
        registry = DatasetRegistry(manager.config.dataset_registry_dir)
        return registry.list_all()

    @app.post(
        "/v1/workspaces/{workspace_id}/datasets",
        status_code=200,
        tags=["Datasets"]
    )
    async def prepare_datasets(
        workspace_id: str,
        request: PrepareDatasetsRequest,
        manager=Depends(get_manager)
    ):
        """
        Prepare datasets in workspace

        Copies or links datasets to workspace/data/ directory.
        """
        logger.info(
            f"Preparing datasets {request.datasets} in workspace: {workspace_id}"
        )

        # Validate workspace exists
        workspace = await manager.get_workspace(workspace_id)

        # Validate all datasets exist in registry
        dataset_registry = Path(manager.config.dataset_registry_dir)
        for dataset_name in request.datasets:
            # Validate dataset_name against path traversal
            validate_path_component(dataset_name, "dataset_name")
            dataset_path = dataset_registry / dataset_name
            if not dataset_path.exists():
                raise DatasetNotFoundError(dataset_name=dataset_name)

        # Prepare datasets through manager
        await manager._prepare_datasets(request.datasets, workspace, request.strategy)

        return {"status": "prepared", "datasets": request.datasets}

    @app.get(
        "/v1/workspaces/{workspace_id}/datasets",
        response_model=List[DatasetInfo],
        tags=["Datasets"]
    )
    async def list_workspace_datasets(
        workspace_id: str,
        manager=Depends(get_manager)
    ):
        """
        List prepared datasets in workspace

        Returns datasets that have been prepared in the workspace.
        """
        logger.info(f"Listing datasets in workspace: {workspace_id}")

        validate_path_component(workspace_id, "workspace_id")
        workspace_service = WorkspaceService(
            base_dir=manager.config.workspace_base_dir,
            dataset_registry=manager.config.dataset_registry_dir,
        )
        return await workspace_service.list_workspace_datasets(workspace_id)

    # =========================================================================
    # Storage Management
    # =========================================================================

    @app.post(
        "/v1/workspaces/{workspace_id}/storage",
        response_model=StorageMountResponse,
        tags=["Storage"]
    )
    async def mount_storage(
        workspace_id: str,
        request: MountStorageRequest,
        manager=Depends(get_manager)
    ):
        """
        Mount storage to a workspace

        Mounts cloud storage (S3, GCS, Azure) to a workspace, making it
        accessible from within the sandbox.
        """
        logger.info(
            f"Mounting storage to workspace: {workspace_id}, "
            f"bucket: {request.storage_config.bucket}"
        )

        # Validate workspace exists
        await manager.get_workspace(workspace_id)

        # Mount storage through manager
        result = await manager.mount_storage(
            workspace_id=workspace_id,
            storage_config=request.storage_config,
            mount_name=request.mount_name,
        )

        return StorageMountResponse(**result)

    @app.get(
        "/v1/workspaces/{workspace_id}/storage",
        response_model=StorageMountListResponse,
        tags=["Storage"]
    )
    async def list_storage_mounts(
        workspace_id: str,
        manager=Depends(get_manager)
    ):
        """
        List storage mounts for a workspace

        Returns all storage mounts configured for a workspace.
        """
        logger.info(f"Listing storage mounts for workspace: {workspace_id}")

        # Validate workspace exists
        await manager.get_workspace(workspace_id)

        mounts = await manager.list_storage_mounts(workspace_id)
        return StorageMountListResponse(
            mounts=[StorageMountResponse(**m) for m in mounts]
        )

    @app.get(
        "/v1/workspaces/{workspace_id}/storage/{mount_name}",
        response_model=StorageMountResponse,
        tags=["Storage"]
    )
    async def get_storage_mount(
        workspace_id: str,
        mount_name: str,
        manager=Depends(get_manager)
    ):
        """
        Get storage mount for a workspace

        Returns configuration for a specific storage mount.
        """
        logger.info(f"Getting storage mount '{mount_name}' for workspace: {workspace_id}")

        # Validate mount_name
        validate_path_component(mount_name, "mount_name")

        result = await manager.get_storage_mount(workspace_id, mount_name)
        return StorageMountResponse(**result)

    @app.delete(
        "/v1/workspaces/{workspace_id}/storage/{mount_name}",
        tags=["Storage"]
    )
    async def unmount_storage(
        workspace_id: str,
        mount_name: str,
        manager=Depends(get_manager)
    ):
        """
        Unmount storage from a workspace

        Removes a storage mount from a workspace.
        """
        logger.info(f"Unmounting storage '{mount_name}' from workspace: {workspace_id}")

        # Validate mount_name
        validate_path_component(mount_name, "mount_name")

        result = await manager.unmount_storage(workspace_id, mount_name)
        return result

    # =========================================================================
    # Template Management
    # =========================================================================

    @app.post(
        "/v1/templates",
        response_model=TemplateResponse,
        status_code=201,
        tags=["Templates"]
    )
    async def build_template(
        request: BuildTemplateRequest,
        manager=Depends(get_manager)
    ):
        """
        Build a new template

        Creates and stores a template configuration.
        """
        logger.info(f"Building template: {request.template.id}")

        template = await manager.build_template(
            template=request.template,
            alias=request.alias,
            wait_timeout=request.wait_timeout,
            debug=request.debug,
        )

        logger.info(f"Template built: {template.id}")
        return TemplateResponse(template=template)

    @app.get("/v1/templates", response_model=TemplateListResponse, tags=["Templates"])
    async def list_templates(manager=Depends(get_manager)):
        """
        List all available templates

        Returns all registered templates.
        """
        logger.info("Listing templates")
        templates = await manager.list_templates()
        return TemplateListResponse(templates=templates)

    @app.get("/v1/templates/{template_id}", response_model=TemplateResponse, tags=["Templates"])
    async def get_template(
        template_id: str,
        manager=Depends(get_manager)
    ):
        """
        Get template information

        Returns detailed information about a specific template.
        """
        logger.info(f"Getting template: {template_id}")

        template = await manager.get_template(template_id)
        return TemplateResponse(template=template)

    @app.delete(
        "/v1/templates/{template_id}",
        status_code=204,
        tags=["Templates"]
    )
    async def delete_template(
        template_id: str,
        manager=Depends(get_manager)
    ):
        """
        Delete a template

        Removes a template and its aliases.
        """
        logger.info(f"Deleting template: {template_id}")

        await manager.delete_template(template_id)
        logger.info(f"Template deleted: {template_id}")

    # =========================================================================
    # Code Execution (Core Functionality)
    # =========================================================================

    @app.post(
        "/v1/workspaces/{workspace_id}/run",
        response_model=ExecutionInfo,
        status_code=201,
        tags=["Execution"]
    )
    async def execute_code(
        workspace_id: str,
        request: ExecuteCodeRequest,
        manager=Depends(get_manager)
    ):
        """
        Execute code in workspace

        This is the main entry point for code execution.

        Flow:
        1. Validate workspace exists
        2. Scan code for security (if enabled)
        3. Decide isolation level
        4. Prepare datasets
        5. Mount workspace
        6. Execute code
        7. Return ExecutionInfo with execution_id
        """
        logger.info(
            f"Executing code in workspace: {workspace_id}, "
            f"mode: {request.mode}"
        )

        # Create ExecutionRequest with workspace_id from path
        execution_request = ExecutionRequest(
            **request.model_dump(),
            workspace_id=workspace_id,
        )

        # Pass the ExecutionRequest to manager
        result = await manager.execute(request=execution_request)

        return ExecutionInfo(
            execution_id=result.execution_id,
            workspace_id=workspace_id,
            status="completed" if result.success else "failed",
        )

    @app.get(
        "/v1/workspaces/{workspace_id}/runs/{execution_id}",
        response_model=ExecutionStatus,
        tags=["Execution"]
    )
    async def get_execution_status(
        workspace_id: str,
        execution_id: str,
        manager=Depends(get_manager)
    ):
        """
        Get execution status

        Returns the current status of an execution.
        """
        logger.info(
            f"Getting execution status: workspace={workspace_id}, "
            f"execution={execution_id}"
        )

        status = await manager.get_execution_status(workspace_id, execution_id)
        return ExecutionStatus(
            execution_id=status["execution_id"],
            workspace_id=status["workspace_id"],
            status=status["status"],
            backend=status.get("backend"),
            created_at=status["created_at"],
            started_at=status.get("started_at"),
            completed_at=status.get("completed_at"),
        )

    @app.post(
        "/v1/workspaces/{workspace_id}/runs/{execution_id}/stop",
        tags=["Execution"]
    )
    async def stop_execution(
        workspace_id: str,
        execution_id: str,
        manager=Depends(get_manager)
    ):
        """
        Stop a running execution

        Terminates an executing task.
        """
        logger.info(
            f"Stopping execution: workspace={workspace_id}, "
            f"execution={execution_id}"
        )

        stopped = await manager.stop_execution(workspace_id, execution_id)
        if stopped:
            return {"status": "stopped", "message": "Execution stopped successfully"}
        else:
            return {"status": "stopped", "message": "Execution was not running"}

    @app.get(
        "/v1/workspaces/{workspace_id}/runs/{execution_id}/logs",
        response_model=ExecutionLogs,
        tags=["Execution"]
    )
    async def get_execution_logs(
        workspace_id: str,
        execution_id: str,
        offset: int = 0,
        limit: int = 1000,
        manager=Depends(get_manager)
    ):
        """
        Get execution logs

        Returns logs from an execution with optional pagination.
        """
        logger.info(
            f"Getting logs: workspace={workspace_id}, "
            f"execution={execution_id}, offset={offset}, limit={limit}"
        )
        # Get execution status which includes stored stdout/stderr
        status = await manager.get_execution_status(workspace_id, execution_id)
        stdout = status.get("stdout", "") or ""
        # Apply pagination
        full_logs = stdout
        paginated_logs = full_logs[offset:offset + limit]
        return ExecutionLogs(
            execution_id=execution_id,
            logs=paginated_logs,
            offset=offset,
            limit=limit,
        )

    # =========================================================================
    # Root Endpoint
    # =========================================================================

    @app.get("/", tags=["Root"])
    async def root():
        """
        API root endpoint

        Returns basic API information.
        """
        return {
            "name": "ds-sandbox API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/v1/health",
        }


# =============================================================================
# Application Instance
# =============================================================================

# Create default application instance
app = create_app()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """CLI entry point for ds-sandbox-api command."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(
        description="ds-sandbox API server",
        prog="ds-sandbox-api"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (default: from DS_SANDBOX_API_HOST env or 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from DS_SANDBOX_API_PORT env or 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level"
    )

    args = parser.parse_args()

    config = SandboxConfig.from_env()

    uvicorn.run(
        "ds_sandbox.api.rest:app",
        host=args.host or config.api_host,
        port=args.port or config.api_port,
        log_level=args.log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
