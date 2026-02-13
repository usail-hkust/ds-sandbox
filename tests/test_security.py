"""
Unit tests for security policies.

Tests the security policy engine including network policies,
resource limits, and security context management.
"""

import pytest
from typing import Any

from ds_sandbox.security.policies import (
    SecurityContext,
    NetworkPolicy,
    ResourceLimits,
    CgroupConfig,
    NetworkWhitelistValidator,
    IsolationLevel,
)


class TestIsolationLevel:
    """Tests for IsolationLevel enum."""

    def test_fast_isolation(self):
        """Test FAST isolation level."""
        assert IsolationLevel.FAST == "fast"

    def test_secure_isolation(self):
        """Test SECURE isolation level."""
        assert IsolationLevel.SECURE == "secure"

    def test_disabled_isolation(self):
        """Test DISABLED isolation level."""
        assert IsolationLevel.DISABLED == "disabled"


class TestNetworkPolicy:
    """Tests for NetworkPolicy model."""

    def test_default_policy(self):
        """Test default network policy."""
        policy = NetworkPolicy()

        assert policy.mode == "disabled"
        assert policy.block_private is True
        assert policy.block_localhost is True
        assert len(policy.whitelist) == 0

    def test_custom_policy(self):
        """Test custom network policy."""
        policy = NetworkPolicy(
            mode="whitelist",
            whitelist=["api.example.com", "*.trusted.com"],
            block_private=False,
            block_localhost=False,
        )

        assert policy.mode == "whitelist"
        assert len(policy.whitelist) == 2

    def test_is_network_enabled(self):
        """Test network enabled check."""
        disabled_policy = NetworkPolicy(mode="disabled")
        whitelist_policy = NetworkPolicy(mode="whitelist")
        proxy_policy = NetworkPolicy(mode="proxy")

        assert disabled_policy.is_network_enabled() is False
        assert whitelist_policy.is_network_enabled() is True
        assert proxy_policy.is_network_enabled() is True

    def test_allows_outbound_disabled(self):
        """Test outbound blocked when disabled."""
        policy = NetworkPolicy(mode="disabled")

        assert policy.allows_outbound("example.com", 80) is False

    def test_allows_outbound_whitelist(self):
        """Test whitelist allows matched hosts."""
        policy = NetworkPolicy(
            mode="whitelist",
            whitelist=["api.example.com"],
        )

        assert policy.allows_outbound("api.example.com", 80) is True
        assert policy.allows_outbound("other.example.com", 80) is False

    def test_allows_outbound_proxy(self):
        """Test proxy mode allows all outbound."""
        policy = NetworkPolicy(mode="proxy")

        assert policy.allows_outbound("example.com", 80) is True
        assert policy.allows_outbound("anything.com", 443) is True

    def test_get_effective_dns(self):
        """Test DNS server selection."""
        policy = NetworkPolicy()
        dns = policy.get_effective_dns()

        assert len(dns) > 0
        assert all(isinstance(d, str) for d in dns)


class TestResourceLimits:
    """Tests for ResourceLimits model."""

    def test_default_limits(self):
        """Test default resource limits."""
        limits = ResourceLimits()

        assert limits.memory_mb == 4096
        assert limits.cpu_cores == 2.0
        assert limits.swap_mb == 0
        assert limits.pids_limit == 1024

    def test_custom_limits(self):
        """Test custom resource limits."""
        limits = ResourceLimits(
            memory_mb=8192,
            cpu_cores=4.0,
            swap_mb=2048,
            pids_limit=2048,
        )

        assert limits.memory_mb == 8192
        assert limits.cpu_cores == 4.0
        assert limits.swap_mb == 2048
        assert limits.pids_limit == 2048

    def test_from_execution_mode_safe(self):
        """Test resource limits for safe mode."""
        limits = ResourceLimits.from_execution_mode("safe")

        assert limits.memory_mb == 2048
        assert limits.cpu_cores == 1.0
        assert limits.pids_limit == 512

    def test_from_execution_mode_fast(self):
        """Test resource limits for fast mode."""
        limits = ResourceLimits.from_execution_mode("fast")

        assert limits.memory_mb == 4096
        assert limits.cpu_cores == 2.0
        assert limits.pids_limit == 1024

    def test_from_execution_mode_secure(self):
        """Test resource limits for secure mode."""
        limits = ResourceLimits.from_execution_mode("secure")

        assert limits.memory_mb == 8192
        assert limits.cpu_cores == 4.0
        assert limits.pids_limit == 2048

    def test_to_cgroup_config(self):
        """Test conversion to cgroup config."""
        limits = ResourceLimits(
            memory_mb=4096,
            cpu_cores=2.0,
            pids_limit=1024,
        )

        config = limits.to_cgroup_config()

        assert isinstance(config, CgroupConfig)
        assert config.memory_limit == 4096
        assert config.cpu_cores == 2.0
        assert config.pids_limit == 1024


class TestCgroupConfig:
    """Tests for CgroupConfig model."""

    def test_default_config(self):
        """Test default cgroup config."""
        config = CgroupConfig()

        assert config.memory_limit == 4096
        assert config.cpu_cores == 2.0
        assert config.pids_limit == 1024

    def test_to_cgroup_spec(self):
        """Test generating cgroup specification."""
        config = CgroupConfig(
            memory_limit=4096,
            cpu_cores=2.0,
            blkio_weight=500,
        )

        spec = config.to_cgroup_spec()

        assert "memory.limit" in spec
        assert "cpu.cores" in spec
        assert spec["cpu.cores"] == "2"

    def test_get_docker_config(self):
        """Test generating Docker config."""
        config = CgroupConfig(
            memory_limit=4096,
            cpu_cores=2.0,
            pids_limit=1024,
        )

        docker_config = config.get_docker_config()

        assert "memory" in docker_config
        assert "cpus" in docker_config
        assert "pids_limit" in docker_config


class TestSecurityContext:
    """Tests for SecurityContext model."""

    def test_default_context(self):
        """Test default security context."""
        context = SecurityContext()

        assert context.network_policy.mode == "disabled"
        assert context.resource_limits.memory_mb == 4096
        assert context.isolation_level == "auto"
        assert context.user == 1000

    def test_custom_context(self):
        """Test custom security context."""
        context = SecurityContext(
            network_policy=NetworkPolicy(mode="whitelist"),
            resource_limits=ResourceLimits(memory_mb=8192),
            isolation_level="secure",
            user=0,
        )

        assert context.network_policy.mode == "whitelist"
        assert context.resource_limits.memory_mb == 8192
        assert context.isolation_level == "secure"
        assert context.user == 0

    def test_get_effective_isolation_auto(self):
        """Test automatic isolation level selection."""
        context = SecurityContext(
            isolation_level="auto",
            network_policy=NetworkPolicy(mode="disabled"),
        )

        isolation = context.get_effective_isolation()

        assert isolation == IsolationLevel.FAST

    def test_get_effective_isolation_whitelist(self):
        """Test isolation for whitelist network."""
        context = SecurityContext(
            isolation_level="auto",
            network_policy=NetworkPolicy(mode="whitelist"),
        )

        isolation = context.get_effective_isolation()

        assert isolation == IsolationLevel.SECURE

    def test_get_effective_isolation_explicit(self):
        """Test explicit isolation level overrides."""
        context = SecurityContext(
            isolation_level="secure",
        )

        isolation = context.get_effective_isolation()

        assert isolation == "secure"

    def test_get_recommended_backend(self):
        """Test backend recommendation."""
        fast_context = SecurityContext(
            network_policy=NetworkPolicy(mode="disabled"),
        )

        secure_context = SecurityContext(
            network_policy=NetworkPolicy(mode="whitelist"),
        )

        assert fast_context.get_recommended_backend() == "docker"
        assert secure_context.get_recommended_backend() == "kata"

    def test_requires_secure_backend(self):
        """Test secure backend requirement check."""
        normal_context = SecurityContext(user=1000)
        root_context = SecurityContext(user=0)

        assert normal_context.requires_secure_backend() is False
        assert root_context.requires_secure_backend() is True

    def test_get_docker_security_opts(self):
        """Test Docker security options generation."""
        context = SecurityContext(
            no_new_privileges=True,
            seccomp_profile="unconfined",
        )

        opts = context.get_docker_security_opts()

        assert "no-new-privileges" in opts
        assert any("seccomp" in opt for opt in opts)

    def test_get_capabilities_config(self):
        """Test capabilities configuration."""
        context = SecurityContext(
            capabilities_add=["CAP_NET_BIND_SERVICE"],
            capabilities_drop=["CAP_SYS_ADMIN"],
        )

        add, drop = context.get_capabilities_config()

        assert "CAP_NET_BIND_SERVICE" in add
        assert "all" in drop
        assert "CAP_SYS_ADMIN" in drop


class TestNetworkWhitelistValidator:
    """Tests for NetworkWhitelistValidator."""

    def test_matches_exact_domain(self):
        """Test exact domain matching."""
        whitelist = ["api.example.com"]

        assert NetworkWhitelistValidator.matches("api.example.com", whitelist) is True
        assert NetworkWhitelistValidator.matches("other.example.com", whitelist) is False

    def test_matches_wildcard(self):
        """Test wildcard domain matching."""
        whitelist = ["*.example.com"]

        assert NetworkWhitelistValidator.matches("api.example.com", whitelist) is True
        assert NetworkWhitelistValidator.matches("test.example.com", whitelist) is True
        assert NetworkWhitelistValidator.matches("example.com", whitelist) is False

    def test_matches_cidr(self):
        """Test CIDR IP range matching."""
        whitelist = ["192.168.1.0/24"]

        assert NetworkWhitelistValidator.matches("192.168.1.100", whitelist) is True
        assert NetworkWhitelistValidator.matches("192.168.2.1", whitelist) is False

    def test_validate_whitelist_valid(self):
        """Test validating valid whitelist."""
        whitelist = ["api.example.com", "*.trusted.com", "10.0.0.0/8"]

        is_valid, errors = NetworkWhitelistValidator.validate_whitelist(whitelist)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_whitelist_invalid(self):
        """Test validating invalid whitelist."""
        whitelist = ["invalidpattern", "*.", "10.0.0.0/99"]

        is_valid, errors = NetworkWhitelistValidator.validate_whitelist(whitelist)

        assert is_valid is False
        assert len(errors) > 0

    def test_normalize_whitelist(self):
        """Test whitelist normalization."""
        whitelist = [
            "https://API.Example.com/",
            "HTTP://other.COM/path",
            " *.test.com ",
        ]

        normalized = NetworkWhitelistValidator.normalize_whitelist(whitelist)

        assert "api.example.com" in normalized
        # "other.com/path" is preserved as-is
        assert any("other" in n for n in normalized)
        assert "*.test.com" in normalized
