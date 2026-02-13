"""
ds-sandbox MCP Server

Model Context Protocol (MCP) server implementation for ds-sandbox.
Provides tools and resources for workspace management and code execution.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

from ..config import SandboxConfig
from ..errors import (
    DatasetNotFoundError,
    WorkspaceNotFoundError,
)
from ..manager import SandboxManager
from ..types import (
    ExecutionRequest,
    ExecutionResult,
)

logger = logging.getLogger(__name__)

# Try to import MCP package, fallback to None if not available
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, Resource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP package not installed. Using standalone implementation.")


@dataclass
class MCPTool:
    """MCP Tool specification"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable


@dataclass
class MCPResource:
    """MCP Resource specification"""
    uri: str
    name: str
    description: str
    getter: Callable


class MCPServer:
    """
    MCP Server for ds-sandbox.

    Implements the Model Context Protocol for workspace management
    and code execution operations.
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize MCP Server.

        Args:
            config: Optional sandbox configuration
        """
        self.config = config or SandboxConfig()
        self.manager = SandboxManager(self.config)
        self._execution_results: Dict[str, ExecutionResult] = {}
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._server = None

        if MCP_AVAILABLE:
            self._server = Server("ds-sandbox-mcp")
            self._setup_handlers()

        self._register_tools()
        self._register_resources()

    def _setup_handlers(self) -> None:
        """Setup MCP server handlers."""
        if not self._server:
            return

        @self._server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.input_schema,
                )
                for tool in self._tools.values()
            ]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name not in self._tools:
                raise ValueError(f"Unknown tool: {name}")

            tool = self._tools[name]
            result = await tool.handler(arguments)

            if isinstance(result, dict):
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            return [TextContent(type="text", text=str(result))]

        @self._server.list_resources()
        async def list_resources() -> List[Resource]:
            return [
                Resource(
                    uri=resource.uri,
                    name=resource.name,
                    description=resource.description,
                )
                for resource in self._resources.values()
            ]

        @self._server.read_resource()
        async def read_resource(uri: str) -> str:
            if uri not in self._resources:
                raise ValueError(f"Unknown resource: {uri}")

            resource = self._resources[uri]
            return await resource.getter()

    def _register_tools(self) -> None:
        """Register all MCP tools."""
        self._tools = {
            "create_workspace": MCPTool(
                name="create_workspace",
                description="Create a new workspace with directory structure. "
                           "The workspace provides isolated storage for code execution.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {
                            "type": "string",
                            "description": "Unique workspace identifier (1-64 chars)",
                            "minLength": 1,
                            "maxLength": 64,
                        },
                        "subdirs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional subdirectories to create",
                            "default": ["data", "models", "outputs"],
                        },
                    },
                    "required": ["workspace_id"],
                },
                handler=self._handle_create_workspace,
            ),
            "delete_workspace": MCPTool(
                name="delete_workspace",
                description="Delete a workspace and all its data. "
                           "This action cannot be undone.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {
                            "type": "string",
                            "description": "Workspace identifier to delete",
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force deletion even if errors occur",
                            "default": False,
                        },
                    },
                    "required": ["workspace_id"],
                },
                handler=self._handle_delete_workspace,
            ),
            "list_workspaces": MCPTool(
                name="list_workspaces",
                description="List all existing workspaces and their status.",
                input_schema={
                    "type": "object",
                    "properties": {},
                },
                handler=self._handle_list_workspaces,
            ),
            "execute_code": MCPTool(
                name="execute_code",
                description="Execute Python code in an isolated sandbox environment. "
                           "Code runs with resource limits and network controls.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        },
                        "workspace_id": {
                            "type": "string",
                            "description": "Workspace ID for execution context",
                        },
                        "datasets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Dataset names to prepare in workspace/data/",
                            "default": [],
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["safe", "fast", "secure", "auto"],
                            "description": "Execution isolation mode",
                            "default": "safe",
                        },
                        "timeout_sec": {
                            "type": "integer",
                            "description": "Timeout in seconds (1-86400)",
                            "default": 3600,
                        },
                        "memory_mb": {
                            "type": "integer",
                            "description": "Memory limit in MB (512-65536)",
                            "default": 4096,
                        },
                        "cpu_cores": {
                            "type": "number",
                            "description": "CPU cores (0.5-16.0)",
                            "default": 2.0,
                        },
                    },
                    "required": ["code", "workspace_id"],
                },
                handler=self._handle_execute_code,
            ),
            "get_execution_result": MCPTool(
                name="get_execution_result",
                description="Get the result of a code execution by execution ID.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "execution_id": {
                            "type": "string",
                            "description": "Execution ID returned from execute_code",
                        },
                    },
                    "required": ["execution_id"],
                },
                handler=self._handle_get_execution_result,
            ),
            "get_workspace": MCPTool(
                name="get_workspace",
                description="Get information about a specific workspace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {
                            "type": "string",
                            "description": "Workspace identifier",
                        },
                    },
                    "required": ["workspace_id"],
                },
                handler=self._handle_get_workspace,
            ),
            "prepare_datasets": MCPTool(
                name="prepare_datasets",
                description="Prepare datasets in a workspace's data directory.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "workspace_id": {
                            "type": "string",
                            "description": "Workspace identifier",
                        },
                        "datasets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Dataset names to prepare",
                        },
                    },
                    "required": ["workspace_id", "datasets"],
                },
                handler=self._handle_prepare_datasets,
            ),
            "list_datasets": MCPTool(
                name="list_datasets",
                description="List all available datasets in the registry.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name_contains": {
                            "type": "string",
                            "description": "Filter by name (case-insensitive)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by tags",
                        },
                        "format_type": {
                            "type": "string",
                            "enum": ["csv", "parquet", "json", "excel", "feather"],
                            "description": "Filter by data format",
                        },
                    },
                },
                handler=self._handle_list_datasets,
            ),
        }

    def _register_resources(self) -> None:
        """Register all MCP resources."""
        self._resources = {
            "workspace_data": MCPResource(
                uri="workspace://{id}/data",
                name="Workspace Data",
                description="Access workspace data directory contents",
                getter=self._get_workspace_data,
            ),
            "dataset_info": MCPResource(
                uri="dataset://{name}",
                name="Dataset Information",
                description="Get information about a registered dataset",
                getter=self._get_dataset_info,
            ),
            "workspace_info": MCPResource(
                uri="workspace://{id}",
                name="Workspace Information",
                description="Get metadata about a workspace",
                getter=self._get_workspace_info,
            ),
        }

    async def _handle_create_workspace(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create_workspace tool call."""
        workspace_id = args["workspace_id"]
        subdirs = args.get("subdirs")

        logger.info(f"Creating workspace: {workspace_id}")

        try:
            workspace = await self.manager.create_workspace(
                workspace_id=workspace_id,
                subdirs=subdirs,
            )
            return {
                "success": True,
                "workspace": workspace.model_dump(),
            }
        except FileExistsError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": "WSP_EXISTS",
            }
        except Exception as e:
            logger.error(f"Failed to create workspace: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_code": "CREATE_FAILED",
            }

    async def _handle_delete_workspace(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle delete_workspace tool call."""
        workspace_id = args["workspace_id"]
        force = args.get("force", False)

        logger.info(f"Deleting workspace: {workspace_id}")

        try:
            await self.manager.delete_workspace(workspace_id, force=force)
            return {
                "success": True,
                "message": f"Workspace '{workspace_id}' deleted",
            }
        except WorkspaceNotFoundError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": "WSP_NOT_FOUND",
            }
        except Exception as e:
            logger.error(f"Failed to delete workspace: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_code": "DELETE_FAILED",
            }

    async def _handle_list_workspaces(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list_workspaces tool call."""
        workspaces = await self.manager.list_workspaces()
        return {
            "workspaces": [w.model_dump() for w in workspaces],
            "count": len(workspaces),
        }

    async def _handle_get_workspace(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_workspace tool call."""
        workspace_id = args["workspace_id"]

        try:
            workspace = await self.manager.get_workspace(workspace_id)
            return {
                "success": True,
                "workspace": workspace.model_dump(),
            }
        except WorkspaceNotFoundError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": "WSP_NOT_FOUND",
            }

    async def _handle_execute_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle execute_code tool call."""
        code = args["code"]
        workspace_id = args["workspace_id"]
        datasets = args.get("datasets", [])
        mode = args.get("mode", "safe")
        timeout_sec = args.get("timeout_sec", 3600)
        memory_mb = args.get("memory_mb", 4096)
        cpu_cores = args.get("cpu_cores", 2.0)

        logger.info(f"Executing code in workspace: {workspace_id}")

        try:
            # Verify workspace exists
            await self.manager.get_workspace(workspace_id)

            # Create execution request
            request = ExecutionRequest(
                code=code,
                workspace_id=workspace_id,
                datasets=datasets,
                mode=mode,
                timeout_sec=timeout_sec,
                memory_mb=memory_mb,
                cpu_cores=cpu_cores,
            )

            # Execute code
            result = await self.manager.execute(request)

            # Store result
            self._execution_results[result.execution_id] = result

            return {
                "success": result.success,
                "execution_id": result.execution_id,
                "workspace_id": result.workspace_id,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
                "artifacts": result.artifacts,
                "backend": result.backend,
                "isolation_level": result.isolation_level,
            }

        except WorkspaceNotFoundError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": "WSP_NOT_FOUND",
            }
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_code": "EXEC_FAILED",
            }

    async def _handle_get_execution_result(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_execution_result tool call."""
        execution_id = args["execution_id"]

        if execution_id not in self._execution_results:
            return {
                "success": False,
                "error": f"Execution '{execution_id}' not found",
                "error_code": "EXEC_NOT_FOUND",
            }

        result = self._execution_results[execution_id]
        return {
            "success": result.success,
            "execution_id": result.execution_id,
            "workspace_id": result.workspace_id,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "artifacts": result.artifacts,
            "backend": result.backend,
            "isolation_level": result.isolation_level,
        }

    async def _handle_prepare_datasets(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prepare_datasets tool call."""
        workspace_id = args["workspace_id"]
        datasets = args["datasets"]

        logger.info(f"Preparing datasets in workspace: {workspace_id}")

        try:
            workspace = await self.manager.get_workspace(workspace_id)
            await self.manager._prepare_datasets(datasets, workspace)

            return {
                "success": True,
                "message": f"Prepared {len(datasets)} datasets",
                "datasets": datasets,
            }
        except WorkspaceNotFoundError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": "WSP_NOT_FOUND",
            }
        except DatasetNotFoundError as e:
            return {
                "success": False,
                "error": str(e),
                "error_code": "DAT_NOT_FOUND",
            }
        except Exception as e:
            logger.error(f"Failed to prepare datasets: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_code": "PREPARE_FAILED",
            }

    async def _handle_list_datasets(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list_datasets tool call."""
        from ..data.registry import DatasetRegistry

        registry = DatasetRegistry(
            registry_dir=self.config.dataset_registry_dir
        )

        datasets = registry.list(
            name_contains=args.get("name_contains"),
            tags=args.get("tags"),
            format_type=args.get("format_type"),
        )

        return {
            "datasets": [d.model_dump() for d in datasets],
            "count": len(datasets),
        }

    async def _get_workspace_data(self, workspace_id: str) -> str:
        """Get workspace data directory contents."""
        try:
            workspace = await self.manager.get_workspace(workspace_id)
            data_path = Path(workspace.host_path) / "data"

            if not data_path.exists():
                return json.dumps({
                    "workspace_id": workspace_id,
                    "data_path": str(data_path),
                    "files": [],
                    "message": "Data directory is empty or does not exist",
                })

            files = []
            for item in data_path.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(data_path)
                    files.append({
                        "name": str(rel_path),
                        "size": item.stat().st_size,
                    })

            return json.dumps({
                "workspace_id": workspace_id,
                "data_path": str(data_path),
                "files": files,
                "count": len(files),
            }, indent=2)
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "workspace_id": workspace_id,
            })

    async def _get_dataset_info(self, dataset_name: str) -> str:
        """Get dataset information."""
        from ..data.registry import DatasetRegistry

        try:
            registry = DatasetRegistry(
                registry_dir=self.config.dataset_registry_dir
            )
            dataset = registry.get(dataset_name)
            return json.dumps(dataset.model_dump(), indent=2)
        except DatasetNotFoundError as e:
            return json.dumps({
                "error": str(e),
                "dataset_name": dataset_name,
            })

    async def _get_workspace_info(self, workspace_id: str) -> str:
        """Get workspace information."""
        try:
            workspace = await self.manager.get_workspace(workspace_id)
            return json.dumps(workspace.model_dump(), indent=2)
        except WorkspaceNotFoundError as e:
            return json.dumps({
                "error": str(e),
                "workspace_id": workspace_id,
            })

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions for MCP protocol."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    def get_resource_definitions(self) -> List[Dict[str, Any]]:
        """Get resource definitions for MCP protocol."""
        return [
            {
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
            }
            for resource in self._resources.values()
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool by name with arguments.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result as dictionary
        """
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        tool = self._tools[name]
        return await tool.handler(arguments)

    async def read_resource(self, uri: str) -> str:
        """
        Read a resource by URI.

        Args:
            uri: Resource URI (e.g., "workspace://{id}" or "dataset://{name}")

        Returns:
            Resource content as string
        """
        # Parse URI
        if uri.startswith("workspace://"):
            # Handle both workspace://{id} and workspace://{id}/data
            path = uri.replace("workspace://", "")
            parts = path.split("/")
            workspace_id = parts[0]
            subresource = parts[1] if len(parts) > 1 else None

            if subresource == "data":
                return await self._get_workspace_data(workspace_id)
            else:
                return await self._get_workspace_info(workspace_id)

        elif uri.startswith("dataset://"):
            dataset_name = uri.replace("dataset://", "")
            return await self._get_dataset_info(dataset_name)

        raise ValueError(f"Unknown resource URI: {uri}")

    async def run(self) -> None:
        """Run the MCP server using stdio transport."""
        if not MCP_AVAILABLE:
            logger.error("MCP package not available. Cannot run server.")
            raise RuntimeError("MCP package is required to run the server.")

        logger.info("Starting MCP server with stdio transport")
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )

    def create_stdio_transport(self):
        """Create stdio transport for the MCP server."""
        if not MCP_AVAILABLE:
            raise RuntimeError("MCP package is required.")

        from mcp.server.stdio import stdio_server
        return stdio_server()


# Standalone JSON-RPC 2.0 implementation for when MCP package is not available
class StandaloneMCPServer:
    """
    Standalone MCP-compatible JSON-RPC 2.0 server.

    Implements a subset of MCP protocol for compatibility with MCP clients.
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize standalone MCP server.

        Args:
            config: Optional sandbox configuration
        """
        self.mcp_server = MCPServer(config)
        logger.info("Initialized standalone MCP server")

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a JSON-RPC 2.0 request.

        Args:
            request: JSON-RPC 2.0 request dict

        Returns:
            JSON-RPC 2.0 response dict
        """
        request_id = request.get("id")
        method = request.get("method")

        # Handle request
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {
                            "list": True,
                            "call": True,
                        },
                        "resources": {
                            "list": True,
                            "read": True,
                        },
                    },
                    "serverInfo": {
                        "name": "ds-sandbox-mcp",
                        "version": "1.0.0",
                    },
                },
            }

        elif method == "tools/list":
            tools = self.mcp_server.get_tool_definitions()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": tools},
            }

        elif method == "tools/call":
            tool_name = request.get("params", {}).get("name")
            arguments = request.get("params", {}).get("arguments", {})

            try:
                result = await self.mcp_server.call_tool(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2),
                            }
                        ],
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": str(e),
                    },
                }

        elif method == "resources/list":
            resources = self.mcp_server.get_resource_definitions()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"resources": resources},
            }

        elif method == "resources/read":
            uri = request.get("params", {}).get("uri")
            try:
                content = await self.mcp_server.read_resource(uri)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "application/json",
                                "text": content,
                            }
                        ],
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32000,
                        "message": str(e),
                    },
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }

    async def run_stdio(self) -> None:
        """Run the server using stdio transport."""
        import sys

        logger.info("Starting standalone MCP server on stdio")

        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                if not line:
                    break

                request = json.loads(line.strip())
                response = await self.handle_request(request)

                if response:
                    print(json.dumps(response), flush=True)

            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {e}",
                    },
                }
                print(json.dumps(error_response), flush=True)

            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32000,
                        "message": str(e),
                    },
                }
                print(json.dumps(error_response), flush=True)


def create_mcp_server(config: Optional[SandboxConfig] = None) -> MCPServer:
    """
    Create an MCP server instance.

    Args:
        config: Optional sandbox configuration

    Returns:
        MCPServer instance
    """
    return MCPServer(config)


def create_standalone_server(config: Optional[SandboxConfig] = None) -> StandaloneMCPServer:
    """
    Create a standalone MCP server (no MCP package required).

    Args:
        config: Optional sandbox configuration

    Returns:
        StandaloneMCPServer instance
    """
    return StandaloneMCPServer(config)


__all__ = [
    "MCPServer",
    "StandaloneMCPServer",
    "create_mcp_server",
    "create_standalone_server",
]
