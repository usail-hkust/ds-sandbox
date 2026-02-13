"""
Docker-based sandbox backend

Uses Docker containers with configurable resource limits for code execution.
Fast startup and good performance for isolated execution.
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import docker
from docker.errors import DockerException, NotFound
from docker.models.containers import Container

from ds_sandbox.backends.base import SandboxBackend
from ds_sandbox.types import ExecutionRequest, ExecutionResult, Workspace


class DockerSandbox(SandboxBackend):
    """
    Docker-based sandbox backend

    Features:
    - Container-based isolation
    - Configurable memory and CPU limits
    - Workspace volume mounting
    - Artifact collection
    - Health checks
    """

    DEFAULT_IMAGE = "python:3.10-slim"
    DEFAULT_MEMORY_MB = 4096
    DEFAULT_CPU_CORES = 2.0
    DEFAULT_TIMEOUT_SEC = 3600

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Docker backend

        Args:
            config: Optional configuration dict with keys:
                - image: Docker image to use (default: python:3.10-slim)
                - memory_mb: Default memory limit in MB
                - cpu_cores: Default CPU cores
                - timeout_sec: Default timeout in seconds
                - network_disabled: Disable network by default
        """
        super().__init__(config)
        self.config = config or {}

        # Configuration with defaults
        self.image = self.config.get("image", self.DEFAULT_IMAGE)
        self.default_memory_mb = self.config.get("memory_mb", self.DEFAULT_MEMORY_MB)
        self.default_cpu_cores = self.config.get("cpu_cores", self.DEFAULT_CPU_CORES)
        self.default_timeout_sec = self.config.get("timeout_sec", self.DEFAULT_TIMEOUT_SEC)
        self.network_disabled = self.config.get("network_disabled", True)

        self._client: Optional[docker.DockerClient] = None

    @property
    def client(self) -> docker.DockerClient:
        """Lazy initialize Docker client"""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def execute(
        self,
        request: ExecutionRequest,
        workspace: Workspace
    ) -> ExecutionResult:
        """
        Execute code in Docker container

        Process:
        1. Pull image if needed
        2. Create and start container
        3. Write code to temporary file in container
        4. Execute code and capture output
        5. Collect artifacts from output directory
        6. Stop and remove container

        Args:
            request: Execution request with code and parameters
            workspace: Workspace information with paths

        Returns:
            ExecutionResult with stdout, stderr, exit code, etc.
        """
        start_time = time.time()
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        workspace_id = workspace.workspace_id

        container: Optional[Container] = None

        try:
            # Resolve resources with defaults
            memory_mb = request.memory_mb or self.default_memory_mb
            cpu_cores = request.cpu_cores or self.default_cpu_cores
            timeout_sec = request.timeout_sec or self.default_timeout_sec

            # Pull image
            await self._pull_image(self.image)

            # Create container with resource limits
            container = await self._create_container(
                workspace=workspace,
                memory_mb=memory_mb,
                cpu_cores=cpu_cores,
                network_policy=request.network_policy,
                network_whitelist=request.network_whitelist,
                allow_internet=request.allow_internet,
                env_vars=request.env_vars
            )

            # Start container
            container.start()
            container_id = container.id
            short_id = container_id[:12]

            # Write code to container and execute
            exec_result = await self._run_code_in_container(
                container=container,
                code=request.code,
                timeout_sec=timeout_sec,
                workspace=workspace
            )

            # Collect artifacts from output directory
            artifacts = await self._collect_artifacts(container, workspace)

            duration_ms = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                success=exec_result["exit_code"] == 0,
                stdout=exec_result["stdout"],
                stderr=exec_result["stderr"],
                exit_code=exec_result["exit_code"],
                duration_ms=duration_ms,
                artifacts=artifacts,
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="docker",
                isolation_level="container",
                metadata={
                    "container_id": short_id,
                    "memory_mb": memory_mb,
                    "cpu_cores": cpu_cores,
                }
            )

        except asyncio.TimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {timeout_sec} seconds",
                exit_code=-1,
                duration_ms=duration_ms,
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="docker",
                isolation_level="container",
                metadata={"error": "timeout"}
            )

        except DockerException as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Docker error: {str(e)}",
                exit_code=-1,
                duration_ms=duration_ms,
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="docker",
                isolation_level="container",
                metadata={"error": str(e)}
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Unexpected error: {str(e)}",
                exit_code=-1,
                duration_ms=duration_ms,
                execution_id=execution_id,
                workspace_id=workspace_id,
                backend="docker",
                isolation_level="container",
                metadata={"error": str(e)}
            )

        finally:
            # Cleanup container
            if container is not None:
                await self._cleanup_container(container)

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Docker backend health

        Returns:
            Dict with status, docker_version, and containers_info
        """
        try:
            # Check Docker API availability
            version_info = self.client.version()
            docker_version = version_info.get("Version", "unknown")

            # Count running containers with our prefix
            containers = self.client.containers.list(
                filters={"name": "ds-sandbox-"},
                all=True
            )
            running_count = len([c for c in containers if c.status == "running"])

            return {
                "status": "healthy",
                "backend": "docker",
                "docker_version": docker_version,
                "api_version": version_info.get("ApiVersion", "unknown"),
                "running_containers": running_count,
                "total_containers": len(containers),
            }

        except DockerException as e:
            return {
                "status": "unhealthy",
                "backend": "docker",
                "error": str(e)
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "docker",
                "error": f"Unexpected error: {str(e)}"
            }

    async def cleanup(self, workspace_id: str) -> None:
        """
        Cleanup Docker resources for a workspace

        Args:
            workspace_id: Workspace ID to cleanup
        """
        container_name = f"ds-sandbox-{workspace_id[:8]}"

        try:
            container = self.client.containers.get(container_name)
            await self._cleanup_container(container)
        except NotFound:
            pass  # Container already removed
        except DockerException as e:
            raise RuntimeError(f"Failed to cleanup container {container_name}: {e}")

    async def _pull_image(self, image: str) -> None:
        """Pull Docker image if not available"""
        try:
            # Check if image exists locally
            self.client.images.get(image)
        except NotFound:
            # Pull image
            await asyncio.to_thread(self.client.images.pull, image)

    async def _create_container(
        self,
        workspace: Workspace,
        memory_mb: int,
        cpu_cores: float,
        network_policy: str,
        network_whitelist: list,
        allow_internet: bool = True,
        env_vars: Optional[Dict[str, str]] = None
    ) -> Container:
        """
        Create Docker container with resource limits

        Args:
            workspace: Workspace information
            memory_mb: Memory limit in MB
            cpu_cores: CPU cores limit
            network_policy: Network policy (allow/deny/whitelist)
            network_whitelist: List of allowed domains/IPs
            allow_internet: Whether to allow internet access
            env_vars: Environment variables

        Returns:
            Docker Container instance (not started)
        """
        container_name = f"ds-sandbox-{workspace.workspace_id[:8]}"

        # Build environment variables
        environment = {
            "WORKSPACE": "/workspace",
            "WORKSPACE_DATA": "/workspace/data",
            "WORKSPACE_MODELS": "/workspace/models",
            "WORKSPACE_OUTPUTS": "/workspace/outputs",
        }
        if env_vars:
            environment.update(env_vars)

        # Build volume bindings
        host_path = workspace.host_path
        volumes = self._build_volume_bindings(host_path, workspace.subdirs)

        # Determine network configuration based on policy
        network_mode, extra_hosts = self._configure_network(
            network_policy=network_policy,
            network_whitelist=network_whitelist,
            allow_internet=allow_internet
        )

        # Container configuration
        mem_limit = f"{memory_mb}m"
        nano_cpus = int(cpu_cores * 1e9)  # Convert to nanocpus for Docker API

        # Create container kwargs
        create_kwargs = {
            "image": self.image,
            "command": ["sleep", "infinity"],  # Keep container running
            "name": container_name,
            "environment": environment,
            "volumes": volumes,
            "mem_limit": mem_limit,
            "nano_cpus": nano_cpus,
            "auto_remove": True,
            "detach": True,
            "working_dir": "/workspace",
        }

        # Handle network configuration
        if network_mode is None:
            # Network is disabled
            create_kwargs["network_disabled"] = True
        else:
            create_kwargs["network"] = network_mode
            if extra_hosts:
                create_kwargs["extra_hosts"] = extra_hosts

        return self.client.containers.create(**create_kwargs)

    def _build_volume_bindings(
        self,
        host_path: str,
        subdirs: list
    ) -> Dict[str, dict]:
        """
        Build Docker volume bindings

        Args:
            host_path: Host workspace path
            subdirs: Workspace subdirectories

        Returns:
            Dict mapping host paths to container mount config
        """
        # Ensure host_path is absolute
        host_path = str(Path(host_path).resolve())

        bindings = {}

        # Main workspace volume
        bindings[f"{host_path}:/workspace"] = {"mode": "rw"}

        # Subdirectory volumes
        for subdir in subdirs:
            host_subdir = f"{host_path}/{subdir}"
            container_path = f"/workspace/{subdir}"
            bindings[f"{host_subdir}:{container_path}"] = {"mode": "rw"}

        return bindings

    def _configure_network(
        self,
        network_policy: str,
        network_whitelist: list,
        allow_internet: bool = True
    ) -> tuple:
        """
        Configure Docker network based on policy.

        Args:
            network_policy: Network policy (allow, deny, whitelist)
            network_whitelist: List of allowed hostnames/IPs
            allow_internet: Whether to allow internet access

        Returns:
            Tuple of (network_mode, extra_hosts)
            - network_mode: Docker network name or None for network_disabled
            - extra_hosts: List of host:IP mappings or None
        """
        # Check allow_internet first
        if not allow_internet:
            # Internet is disabled entirely
            return (None, None)

        if network_policy == "deny":
            # Explicitly deny network access
            return (None, None)

        if network_policy == "whitelist":
            if not network_whitelist:
                # No whitelist provided, block all network
                return (None, None)

            # Build extra_hosts list from whitelist
            extra_hosts = []
            for host in network_whitelist:
                if ":" in host:
                    # IP:port format, only use IP
                    host = host.split(":")[0]
                if host and host not in extra_hosts:
                    extra_hosts.append(host)

            if not extra_hosts:
                return (None, None)

            # Use bridge network with extra_hosts for whitelist
            return ("bridge", extra_hosts)

        # Default to "allow" - use bridge network
        return ("bridge", None)

    async def _run_code_in_container(
        self,
        container: Container,
        code: str,
        timeout_sec: int,
        workspace: Workspace
    ) -> Dict[str, Any]:
        """
        Run code inside container and capture output

        Args:
            container: Running Docker container
            code: Python code to execute
            timeout_sec: Execution timeout in seconds
            workspace: Workspace information

        Returns:
            Dict with stdout, stderr, exit_code
        """
        # Write code to a temporary file in the container
        script_filename = f"_temp_script_{uuid.uuid4().hex[:8]}.py"
        script_path = f"/workspace/{script_filename}"

        # Write code to file using exec
        write_cmd = f"cat > {script_path} << 'PYTHONEOF'\n{code}\nPYTHONEOF"
        await self._exec_command(container, ["sh", "-c", write_cmd])

        # Execute the script
        exec_cmd = ["python", script_path]
        exec_result = await self._exec_with_timeout(
            container,
            exec_cmd,
            timeout_sec=timeout_sec
        )

        # Clean up temporary script
        cleanup_cmd = ["rm", "-f", script_path]
        try:
            await self._exec_command(container, cleanup_cmd)
        except Exception:
            pass  # Ignore cleanup errors

        return exec_result

    async def _exec_command(
        self,
        container: Container,
        cmd: list,
        env: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Execute command in container and return result

        Args:
            container: Docker container
            command: Command to execute
            env: Optional environment variables

        Returns:
            Dict with stdout, stderr, exit_code
        """
        # Use the exec API
        exec_output = await asyncio.to_thread(
            container.exec_run,
            cmd,
            environment=env,
            workdir="/workspace"
        )

        return {
            "stdout": exec_output.output.decode("utf-8", errors="replace"),
            "stderr": "",
            "exit_code": exec_output.exit_code
        }

    async def _exec_with_timeout(
        self,
        container: Container,
        cmd: list,
        timeout_sec: int
    ) -> Dict[str, Any]:
        """
        Execute command with timeout

        Args:
            container: Docker container
            cmd: Command to execute
            timeout_sec: Timeout in seconds

        Returns:
            Dict with stdout, stderr, exit_code
        """
        try:
            # Use asyncio timeout
            async with asyncio.timeout(timeout_sec):
                return await self._exec_command(container, cmd)

        except asyncio.TimeoutError:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout_sec} seconds",
                "exit_code": -1
            }

    async def _collect_artifacts(
        self,
        container: Container,
        workspace: Workspace
    ) -> list:
        """
        Collect artifacts from container's outputs directory

        Args:
            container: Docker container
            workspace: Workspace information

        Returns:
            List of artifact file paths
        """
        artifacts_dir = "/workspace/outputs"
        artifacts = []

        try:
            # List files in outputs directory
            result = await asyncio.to_thread(
                container.exec_run,
                ["find", artifacts_dir, "-type", "f", "-name", "*"]
            )

            if result.exit_code == 0 and result.output:
                files = result.output.decode("utf-8").strip().split("\n")
                for filepath in files:
                    if filepath and filepath != artifacts_dir:
                        # Get relative path from workspace
                        rel_path = filepath.replace("/workspace/", "")
                        artifacts.append(rel_path)

        except DockerException:
            pass  # No artifacts directory or files

        return artifacts

    async def _cleanup_container(self, container: Container) -> None:
        """
        Safely stop and remove container

        Args:
            container: Docker container to cleanup
        """
        try:
            if container.status == "running":
                await asyncio.to_thread(container.stop, timeout=5)
            await asyncio.to_thread(container.remove, force=True)
        except NotFound:
            pass  # Container already removed
        except DockerException:
            # Log but don't raise - cleanup failures shouldn't break execution
            pass

    async def stop_execution(self, execution_id: str, workspace_id: str) -> bool:
        """
        Stop a running execution by stopping the container.

        Args:
            execution_id: Execution identifier
            workspace_id: Workspace identifier

        Returns:
            True if execution was stopped, False otherwise
        """
        container_name = f"ds-sandbox-{workspace_id[:8]}"

        try:
            container = self.client.containers.get(container_name)
            if container.status == "running":
                await asyncio.to_thread(container.stop, timeout=5)
                logger.info(f"Stopped container {container_name} for execution {execution_id}")
                return True
        except NotFound:
            logger.debug(f"Container {container_name} not found for execution {execution_id}")
        except DockerException as e:
            logger.error(f"Failed to stop container {container_name}: {e}")

        return False

    def close(self) -> None:
        """Close Docker client connection"""
        if self._client is not None:
            self._client.close()
            self._client = None
