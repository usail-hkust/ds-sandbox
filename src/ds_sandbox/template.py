"""
Template builder for ds-sandbox

Provides a fluent API for building custom sandbox templates.
"""

import logging
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ds_sandbox.types import Template

logger = logging.getLogger(__name__)

__all__ = ["TemplateBuilder", "BuildOptions"]


# Default base images for different template types
DEFAULT_E2B_IMAGE = "e2b-dev/awesome-agent-python"
DEFAULT_PYTHON_IMAGE = "python:{version}-slim"
DEFAULT_NODE_IMAGE = "node:{version}-slim"
DEFAULT_UBUNTU_IMAGE = "ubuntu:{version}"


class BuildOptions(BaseModel):
    """Options for building a template"""

    alias: str = Field(..., description="Template alias")
    wait_timeout: int = Field(default=60, description="Wait timeout in seconds")
    debug: bool = Field(default=False, description="Enable debug mode")


class TemplateBuilder:
    """
    Fluent API for building custom sandbox templates.

    Provides a chainable interface for configuring template properties
    including base image, environment variables, files, and commands.

    Example:
        >>> from ds_sandbox.template import TemplateBuilder
        >>>
        >>> template = (TemplateBuilder()
        ...     .from_python_image("3.11")
        ...     .set_envs({"MY_VAR": "value"})
        ...     .copy("setup.sh", "/home/user/setup.sh")
        ...     .run_cmd("pip install numpy pandas")
        ...     .set_workdir("/home/user")
        ...     .build("my-template"))
    """

    def __init__(self):
        """Initialize the template builder with default values"""
        self._id: str = f"template-{uuid.uuid4().hex[:8]}"
        self._name: Optional[str] = None
        self._description: Optional[str] = None
        self._image: Optional[str] = None
        self._env: Dict[str, str] = {}
        self._files: Dict[str, str] = {}
        self._copy_files: List[Dict[str, str]] = []
        self._cmd: List[str] = []
        self._run_cmds: List[str] = []
        self._start_cmd: Optional[str] = None
        self._ready_cmd: Optional[str] = None
        self._ready_timeout: int = 20
        self._user: Optional[str] = None
        self._workdir: Optional[str] = None
        self._cpu_count: int = 2
        self._memory_mb: int = 2048
        self._aliases: List[str] = []
        self._skip_cache: bool = False
        self._metadata: Dict[str, Any] = {}

    def id(self, template_id: str) -> "TemplateBuilder":
        """Set template ID"""
        self._id = template_id
        return self

    def name(self, name: str) -> "TemplateBuilder":
        """Set template name"""
        self._name = name
        return self

    def description(self, description: str) -> "TemplateBuilder":
        """Set template description"""
        self._description = description
        return self

    def from_base_image(self) -> "TemplateBuilder":
        """Use the default E2B base image"""
        self._image = DEFAULT_E2B_IMAGE
        return self

    def from_image(self, image: str) -> "TemplateBuilder":
        """Use a custom Docker image"""
        self._image = image
        return self

    def from_python_image(self, version: str = "3.11") -> "TemplateBuilder":
        """Use a Python Docker image"""
        self._image = DEFAULT_PYTHON_IMAGE.format(version=version)
        return self

    def from_node_image(self, version: str = "20") -> "TemplateBuilder":
        """Use a Node.js Docker image"""
        self._image = DEFAULT_NODE_IMAGE.format(version=version)
        return self

    def from_ubuntu_image(self, version: str = "22.04") -> "TemplateBuilder":
        """Use an Ubuntu Docker image"""
        self._image = DEFAULT_UBUNTU_IMAGE.format(version=version)
        return self

    def from_template(self, template_id: str) -> "TemplateBuilder":
        """Extend an existing template by ID"""
        # This will be resolved during build
        self._metadata = self._metadata or {}
        self._metadata["extends"] = template_id
        return self

    def from_dockerfile(self, dockerfile_content: str) -> "TemplateBuilder":
        """Parse a Dockerfile and extract configuration"""
        lines = dockerfile_content.strip().split("\n")

        for line in lines:
            line = line.strip()

            # Parse FROM
            if line.startswith("FROM "):
                image = line[5:].strip()
                # Handle multi-stage builds - take the first FROM
                if not self._image:
                    self._image = image.split(" AS ")[0].strip()

            # Parse ENV
            elif line.startswith("ENV "):
                env_line = line[4:].strip()
                if "=" in env_line:
                    key, value = env_line.split("=", 1)
                    self._env[key.strip()] = value.strip()

            # Parse WORKDIR
            elif line.startswith("WORKDIR "):
                self._workdir = line[8:].strip()

            # Parse USER
            elif line.startswith("USER "):
                self._user = line[5:].strip()

            # Parse COPY
            elif line.startswith("COPY "):
                parts = line[5:].strip().split()
                if len(parts) >= 2:
                    src, dest = parts[0], parts[1]
                    # Handle wildcards
                    if "*" in src:
                        self._copy_files.append({
                            "src": src,
                            "dest": dest,
                            "is_pattern": True
                        })
                    else:
                        self._copy_files.append({
                            "src": src,
                            "dest": dest
                        })

            # Parse RUN
            elif line.startswith("RUN "):
                cmd = line[4:].strip()
                self._run_cmds.append(cmd)

        return self

    def set_envs(self, env_dict: Dict[str, str]) -> "TemplateBuilder":
        """Set environment variables"""
        self._env.update(env_dict)
        return self

    def set_start_cmd(self, cmd: str, wait_timeout: int = 60) -> "TemplateBuilder":
        """Set start command with wait timeout"""
        self._start_cmd = cmd
        self._metadata = self._metadata or {}
        self._metadata["wait_timeout"] = wait_timeout
        return self

    def set_ready_cmd(self, cmd: str, timeout: int = 20) -> "TemplateBuilder":
        """Set ready command to check if sandbox is ready"""
        self._ready_cmd = cmd
        self._ready_timeout = timeout
        return self

    def copy(self, src: str, dest: str) -> "TemplateBuilder":
        """Add a file to copy into the template"""
        self._copy_files.append({"src": src, "dest": dest})
        return self

    def add_file(self, path: str, content: str) -> "TemplateBuilder":
        """Add a file with content to create in the template"""
        self._files[path] = content
        return self

    def run_cmd(self, cmd: str) -> "TemplateBuilder":
        """Add a command to run during build"""
        self._run_cmds.append(cmd)
        return self

    def set_user(self, user: str) -> "TemplateBuilder":
        """Set default user"""
        self._user = user
        return self

    def set_workdir(self, path: str) -> "TemplateBuilder":
        """Set working directory"""
        self._workdir = path
        return self

    def set_cpu_count(self, count: int) -> "TemplateBuilder":
        """Set CPU count"""
        self._cpu_count = count
        return self

    def set_memory_mb(self, mb: int) -> "TemplateBuilder":
        """Set memory in MB"""
        self._memory_mb = mb
        return self

    def add_alias(self, alias: str) -> "TemplateBuilder":
        """Add an alias for the template"""
        self._aliases.append(alias)
        return self

    def skip_cache(self) -> "TemplateBuilder":
        """Skip cache during build"""
        self._skip_cache = True
        return self

    def build(
        self,
        alias: Optional[str] = None,
        wait_timeout: int = 60,
        debug: bool = False,
    ) -> Template:
        """
        Build the template from the configured options.

        Args:
            alias: Primary alias for the template
            wait_timeout: Wait timeout in seconds
            debug: Enable debug mode

        Returns:
            Template object

        Example:
            >>> template = builder.build("my-template")
            >>> print(template.id)
            my-template
        """
        # Use alias as ID if provided
        template_id = alias or self._id

        # Add primary alias to aliases list
        aliases = list(self._aliases)
        if alias and alias not in aliases:
            aliases.insert(0, alias)

        # Build metadata
        metadata = self._metadata or {}
        metadata["wait_timeout"] = wait_timeout
        metadata["debug"] = debug

        template = Template(
            id=template_id,
            name=self._name or template_id,
            description=self._description,
            image=self._image,
            env=self._env,
            files=self._files,
            cmd=self._cmd,
            start_cmd=self._start_cmd,
            ready_cmd=self._ready_cmd,
            ready_timeout=self._ready_timeout,
            user=self._user,
            workdir=self._workdir,
            cpu_count=self._cpu_count,
            memory_mb=self._memory_mb,
            copy_files=self._copy_files,
            run_cmds=self._run_cmds,
            aliases=aliases,
            skip_cache=self._skip_cache,
            metadata=metadata,
        )

        logger.info(f"Built template: {template_id}")
        return template
