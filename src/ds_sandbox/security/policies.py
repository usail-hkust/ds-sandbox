"""
Security Policy Engine for ds-sandbox.

Provides sandbox isolation and security controls including:
- Network policy enforcement
- Resource limits via cgroups
- Security context management
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import re


__all__ = [
    "NetworkPolicy",
    "ResourceLimits",
    "SecurityContext",
    "CgroupConfig",
    "NetworkWhitelistValidator",
    "IsolationLevel",
]


class IsolationLevel:
    """Isolation level constants"""
    DISABLED = "disabled"       # No isolation (not recommended)
    FAST = "fast"               # Docker-based isolation
    SECURE = "secure"          # Firecracker/Kata VM isolation


class NetworkPolicy(BaseModel):
    """
    Network access policy for sandbox execution.

    Attributes:
        mode: Network policy mode
            - disabled: No network access (complete isolation)
            - whitelist: Only allow access to whitelisted domains
            - proxy: Route through a secure proxy
        whitelist: List of allowed domain patterns
        block_private: Block access to private IP ranges
        block_localhost: Block access to localhost
        dns_servers: Custom DNS servers to use
        outbound_ports: Allowed outbound ports (empty = all)
        inbound_ports: Allowed inbound ports (empty = none)
    """

    mode: Literal["disabled", "whitelist", "proxy"] = Field(
        default="disabled",
        description="Network policy mode"
    )

    whitelist: List[str] = Field(
        default_factory=list,
        description="Domain whitelist patterns (supports wildcards)"
    )

    block_private: bool = Field(
        default=True,
        description="Block access to private IP ranges (10.x, 172.16.x, 192.168.x)"
    )

    block_localhost: bool = Field(
        default=True,
        description="Block access to localhost/127.0.0.1"
    )

    dns_servers: List[str] = Field(
        default_factory=list,
        description="Custom DNS servers (if empty, use system default)"
    )

    outbound_ports: List[int] = Field(
        default_factory=list,
        description="Allowed outbound ports (empty = all ports)"
    )

    inbound_ports: List[int] = Field(
        default_factory=list,
        description="Allowed inbound ports (empty = no inbound)"
    )

    def is_network_enabled(self) -> bool:
        """Check if network access is enabled"""
        return self.mode != "disabled"

    def allows_outbound(self, host: str, port: int) -> bool:
        """Check if outbound connection is allowed"""
        if self.mode == "disabled":
            return False

        if self.outbound_ports and port not in self.outbound_ports:
            return False

        if self.mode == "whitelist":
            return NetworkWhitelistValidator.matches(host, self.whitelist)

        return True

    def allows_inbound(self, port: int) -> bool:
        """Check if inbound connection is allowed"""
        if self.mode == "disabled":
            return False

        if not self.inbound_ports:
            return False

        return port in self.inbound_ports

    def get_effective_dns(self) -> List[str]:
        """Get effective DNS servers"""
        if self.dns_servers:
            return self.dns_servers
        return ["8.8.8.8", "8.8.4.4"]  # Default Google DNS


class ResourceLimits(BaseModel):
    """
    Resource limits enforced via cgroups.

    All memory values are in MB, time values in seconds.

    Attributes:
        memory_mb: Maximum memory usage
        swap_mb: Maximum swap usage (0 = disabled)
        cpu_cores: CPU cores to allocate (can be fractional)
        cpu_shares: CPU shares (1024 = normal priority)
        blkio_weight: Block I/O weight (default 500)
        pids_limit: Maximum number of processes (0 = unlimited)
        fsize_limit_kb: Maximum file size in KB (0 = unlimited)
        nofile_limit: Maximum number of open files (0 = unlimited)
        core_limit_kb: Maximum core dump size in KB (0 = disabled)
    """

    memory_mb: int = Field(
        default=4096,
        ge=128,
        le=65536,
        description="Memory limit in MB"
    )

    swap_mb: int = Field(
        default=0,
        ge=0,
        le=65536,
        description="Swap limit in MB (0 = disabled)"
    )

    cpu_cores: float = Field(
        default=2.0,
        ge=0.5,
        le=64.0,
        description="Number of CPU cores"
    )

    cpu_shares: int = Field(
        default=1024,
        ge=0,
        le=10240,
        description="CPU shares (1024 = normal priority)"
    )

    blkio_weight: int = Field(
        default=500,
        ge=10,
        le=1000,
        description="Block I/O weight"
    )

    pids_limit: int = Field(
        default=1024,
        ge=0,
        le=100000,
        description="Maximum processes (0 = unlimited)"
    )

    fsize_limit_kb: int = Field(
        default=0,
        ge=0,
        description="Maximum file size in KB (0 = unlimited)"
    )

    nofile_limit: int = Field(
        default=1048576,
        ge=0,
        description="Maximum open files (0 = unlimited)"
    )

    core_limit_kb: int = Field(
        default=0,
        ge=0,
        description="Core dump size in KB (0 = disabled)"
    )

    def to_cgroup_config(self) -> "CgroupConfig":
        """Convert to cgroup configuration"""
        return CgroupConfig(
            memory_limit=self.memory_mb,
            memory_swap=self.swap_mb,
            cpu_cores=self.cpu_cores,
            cpu_shares=self.cpu_shares,
            blkio_weight=self.blkio_weight,
            pids_limit=self.pids_limit,
            fsize_limit_kb=self.fsize_limit_kb,
            nofile_limit=self.nofile_limit,
            core_limit_kb=self.core_limit_kb,
        )

    @classmethod
    def from_execution_mode(
        cls,
        mode: Literal["safe", "fast", "secure"],
        custom_limits: Optional["ResourceLimits"] = None,
    ) -> "ResourceLimits":
        """
        Create resource limits from execution mode.

        Args:
            mode: Execution mode
            custom_limits: Custom overrides (if any)

        Returns:
            ResourceLimits instance
        """
        if custom_limits:
            return custom_limits

        presets = {
            "safe": cls(memory_mb=2048, cpu_cores=1.0, pids_limit=512),
            "fast": cls(memory_mb=4096, cpu_cores=2.0, pids_limit=1024),
            "secure": cls(memory_mb=8192, cpu_cores=4.0, pids_limit=2048),
        }

        return presets.get(mode, presets["fast"])


@dataclass
class CgroupConfig:
    """
    Low-level cgroup configuration.

    This class represents the actual cgroup settings that will be
    applied to the container/VM.
    """

    memory_limit: int = 4096  # MB
    memory_swap: int = 0  # MB
    cpu_cores: float = 2.0
    cpu_shares: int = 1024
    blkio_weight: int = 500
    pids_limit: int = 1024
    fsize_limit_kb: int = 0
    nofile_limit: int = 1048576
    core_limit_kb: int = 0

    def to_cgroup_spec(self) -> Dict[str, str]:
        """Generate cgroup specification for container creation"""
        spec = {
            "memory.limit": f"{self.memory_limit}M",
            "memory.memsw.limit": f"{self.memory_swap}M" if self.memory_swap else "max",
            "memory.swappiness": "0",
            "cpu.cores": str(int(self.cpu_cores)),
            "cpu.shares": str(self.cpu_shares),
            "blkio.weight": str(self.blkio_weight),
            "pids.max": str(self.pids_limit) if self.pids_limit else "max",
        }

        if self.fsize_limit_kb:
            spec["fsize.limit"] = f"{self.fsize_limit_kb}K"

        if self.nofile_limit:
            spec["nofile.limit"] = str(self.nofile_limit)

        if self.core_limit_kb:
            spec["core.limit"] = f"{self.core_limit_kb}K"

        return spec

    def get_docker_config(self) -> Dict[str, Any]:
        """Generate Docker security options"""
        return {
            "memory": f"{self.memory_limit}M",
            "memswap": f"{self.memory_swap}M" if self.memory_swap else "",
            "cpus": self.cpu_cores,
            "pids_limit": self.pids_limit if self.pids_limit else -1,
            "ulimits": self._get_docker_ulimits(),
        }

    def _get_docker_ulimits(self) -> List[Dict[str, Any]]:
        """Generate Docker ulimits configuration"""
        ulimits = []

        if self.fsize_limit_kb:
            ulimits.append({
                "Name": "fsize",
                "Soft": self.fsize_limit_kb,
                "Hard": self.fsize_limit_kb,
            })

        if self.nofile_limit:
            ulimits.append({
                "Name": "nofile",
                "Soft": self.nofile_limit,
                "Hard": self.nofile_limit,
            })

        if self.core_limit_kb:
            ulimits.append({
                "Name": "core",
                "Soft": self.core_limit_kb,
                "Hard": self.core_limit_kb,
            })

        return ulimits


class SecurityContext(BaseModel):
    """
    Security context for sandbox execution.

    Combines network policy and resource limits, and provides
    automatic selection of isolation level based on security
    requirements.

    Attributes:
        network_policy: Network access policy
        resource_limits: Resource limits configuration
        isolation_level: Requested isolation level (auto = automatic selection)
        user: User ID to run as (0 = root)
        read_only_rootfs: Make root filesystem read-only
        no_new_privileges: Prevent privilege escalation
        allow_privilege_escalation: Allow privilege escalation
        capabilities: Linux capabilities to add/drop
        seccomp_profile: Seccomp profile path or name
        apparmor_profile: AppArmor profile path or name
        run_as_username: Username to run as (alternative to user ID)
    """

    network_policy: NetworkPolicy = Field(
        default_factory=NetworkPolicy,
        description="Network access policy"
    )

    resource_limits: ResourceLimits = Field(
        default_factory=ResourceLimits,
        description="Resource limits"
    )

    isolation_level: Literal["auto", "fast", "secure"] = Field(
        default="auto",
        description="Isolation level (auto = automatic selection)"
    )

    user: int = Field(
        default=1000,
        ge=0,
        description="User ID to run as"
    )

    read_only_rootfs: bool = Field(
        default=True,
        description="Make root filesystem read-only"
    )

    no_new_privileges: bool = Field(
        default=True,
        description="Prevent gaining new privileges"
    )

    allow_privilege_escalation: bool = Field(
        default=False,
        description="Allow privilege escalation"
    )

    capabilities_add: List[str] = Field(
        default_factory=list,
        description="Linux capabilities to add"
    )

    capabilities_drop: List[str] = Field(
        default_factory=list,
        description="Linux capabilities to drop"
    )

    seccomp_profile: Optional[str] = Field(
        default=None,
        description="Seccomp profile (path or 'unconfined')"
    )

    apparmor_profile: Optional[str] = Field(
        default=None,
        description="AppArmor profile (path or 'unconfined')"
    )

    run_as_username: Optional[str] = Field(
        default=None,
        description="Username to run as (alternative to user ID)"
    )

    def get_effective_isolation(self, code_scan_result: Optional[Any] = None) -> str:
        """
        Get effective isolation level based on context.

        Args:
            code_scan_result: Optional code scan result for risk assessment

        Returns:
            Isolation level string
        """
        if self.isolation_level != "auto":
            return self.isolation_level

        # If no network access needed, use fast isolation
        if not self.network_policy.is_network_enabled():
            return IsolationLevel.FAST

        # If whitelisted network access, use secure isolation
        if self.network_policy.mode == "whitelist":
            return IsolationLevel.SECURE

        # Default to fast isolation
        return IsolationLevel.FAST

    def get_recommended_backend(self) -> str:
        """Get recommended backend based on security context"""
        isolation = self.get_effective_isolation()

        if isolation == IsolationLevel.SECURE:
            return "kata"  # Highest isolation
        elif isolation == IsolationLevel.FAST:
            return "docker"  # Good balance of security and performance
        else:
            return "docker"  # Fallback

    def requires_secure_backend(self) -> bool:
        """Check if secure backend is required"""
        return (
            self.network_policy.mode == "whitelist"
            or self.user == 0
            or len(self.capabilities_add) > 0
        )

    def get_docker_security_opts(self) -> List[str]:
        """Generate Docker security options"""
        opts = []

        if self.no_new_privileges:
            opts.append("no-new-privileges")

        if self.seccomp_profile:
            if self.seccomp_profile != "unconfined":
                opts.append(f"seccomp={self.seccomp_profile}")
            else:
                opts.append("seccomp=unconfined")

        if self.apparmor_profile:
            if self.apparmor_profile != "unconfined":
                opts.append(f"apparmor={self.apparmor_profile}")
            else:
                opts.append("apparmor=unconfined")

        return opts

    def get_capabilities_config(self) -> Tuple[List[str], List[str]]:
        """Get capabilities configuration (drop all, add specific)"""
        # Always drop all capabilities by default
        drop = ["all"] + self.capabilities_drop

        # Add only what's explicitly requested
        add = self.capabilities_add if self.capabilities_add else []

        return (add, drop)


class NetworkWhitelistValidator:
    """
    Validates network whitelist patterns.

    Supports:
    - Exact domain matching (example.com)
    - Wildcard matching (*.example.com)
    - CIDR notation for IP addresses (192.168.1.0/24)
    - Port specification (example.com:443)
    """

    # Regex patterns
    DOMAIN_PATTERN = re.compile(
        r"^(?:[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
    )
    WILDCARD_PATTERN = re.compile(
        r"^\*(?:\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
    )
    IP_CIDR_PATTERN = re.compile(
        r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
        r"(?:/(?:[0-9]|[12][0-9]|3[0-2]))?$"
    )

    @classmethod
    def matches(cls, host: str, whitelist: List[str]) -> bool:
        """
        Check if host matches any pattern in whitelist.

        Args:
            host: Hostname or IP address to check
            whitelist: List of whitelist patterns

        Returns:
            True if host matches any pattern
        """
        # Extract hostname and port if present
        hostname, port = cls._parse_host(host)

        for pattern in whitelist:
            if cls._matches_pattern(hostname, pattern):
                return True

        return False

    @classmethod
    def _parse_host(cls, host: str) -> Tuple[str, Optional[int]]:
        """Parse host into hostname and port"""
        if ":" in host:
            hostname, port_str = host.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                port = None
            return hostname, port
        return host, None

    @classmethod
    def _matches_pattern(cls, host: str, pattern: str) -> bool:
        """Check if host matches a single pattern"""
        # Parse pattern
        pattern_host, pattern_port = cls._parse_host(pattern)

        # Check port if specified in pattern
        if pattern_port is not None:
            # Port-specific match not implemented yet
            pass

        # Check wildcard pattern
        if cls.WILDCARD_PATTERN.match(pattern_host):
            suffix = pattern_host[1:]  # Remove leading *
            return host.endswith(suffix)

        # Check domain pattern
        if cls.DOMAIN_PATTERN.match(pattern_host):
            # Exact match or subdomain
            return host == pattern_host or host.endswith("." + pattern_host)

        # Check CIDR pattern
        if "/" in pattern_host:
            return cls._matches_cidr(host, pattern_host)

        # Fallback: exact string match
        return host == pattern_host

    @classmethod
    def _matches_cidr(cls, ip: str, cidr: str) -> bool:
        """Check if IP matches CIDR range"""
        import ipaddress

        try:
            network = ipaddress.ip_network(cidr, strict=False)
            return ipaddress.ip_address(ip) in network
        except ValueError:
            return False

    @classmethod
    def validate_whitelist(cls, whitelist: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate whitelist patterns.

        Args:
            whitelist: List of patterns to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        for pattern in whitelist:
            if not pattern:
                errors.append("Empty pattern in whitelist")
                continue

            pattern_host, _ = cls._parse_host(pattern)

            # Check for valid patterns
            is_valid = (
                cls.WILDCARD_PATTERN.match(pattern_host)
                or cls.DOMAIN_PATTERN.match(pattern_host)
                or cls.IP_CIDR_PATTERN.match(pattern_host)
            )

            if not is_valid:
                errors.append(f"Invalid pattern: {pattern}")

        return (len(errors) == 0, errors)

    @classmethod
    def normalize_whitelist(cls, whitelist: List[str]) -> List[str]:
        """
        Normalize whitelist patterns.

        Args:
            whitelist: List of patterns to normalize

        Returns:
            Normalized list of patterns
        """
        normalized = []

        for pattern in whitelist:
            pattern = pattern.strip().lower()

            # Remove protocol if present
            if pattern.startswith(("http://", "https://")):
                pattern = pattern.split("://", 1)[1]

            # Remove trailing slash
            pattern = pattern.rstrip("/")

            if pattern:
                normalized.append(pattern)

        return normalized
