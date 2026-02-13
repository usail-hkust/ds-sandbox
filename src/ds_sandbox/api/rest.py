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
        )

        logger.info(f"Workspace created: {workspace.workspace_id}")
        return workspace

    @app.get("/v1/workspaces", response_model=List[Workspace], tags=["Workspaces"])
    async def list_workspaces(manager=Depends(get_manager)):
        """
        List all workspaces

        Returns a list of all available workspaces.
        """
        logger.info("Listing workspaces")
        return await manager.list_workspaces()

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
