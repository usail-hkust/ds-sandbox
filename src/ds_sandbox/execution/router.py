"""
Isolation router functionality for the sandbox manager.

This module provides the IsolationRouter class for deciding
which backend to use for code execution.
"""

import logging
from typing import List, Optional

from ds_sandbox.config import SandboxConfig
from ds_sandbox.types import CodeScanResult, ExecutionRequest
from ds_sandbox.errors import InvalidRequestError

logger = logging.getLogger(__name__)


class IsolationRouter:
    """
    Isolation level router - decides which backend to use.

    Routing logic:
    1. Explicit backend selection via config.default_backend=local
    2. GPU or restricted network access -> Docker
    3. Code risk score -> auto routing
    4. Default -> config.default_backend (with auto fallback to Docker)
    """

    # Mapping from mode to backend
    MODE_BACKEND_MAP = {
        "secure": "docker",
        "fast": "docker",
        "safe": "docker",
    }

    # Risk score thresholds
    HIGH_RISK_THRESHOLD = 0.7
    MEDIUM_RISK_THRESHOLD = 0.3

    def __init__(self, config: SandboxConfig):
        """
        Initialize isolation router.

        Args:
            config: Sandbox configuration
        """
        self.config = config

    def decide_backend(
        self,
        request: ExecutionRequest,
        code_scan_result: Optional[CodeScanResult] = None,
    ) -> str:
        """
        Decide which backend to use for execution.

        Args:
            request: Execution request with all parameters
            code_scan_result: Optional code security scan result

        Returns:
            Backend name to use (docker, firecracker, kata, etc.)
        """
        # Step 0: Manual override for local backend.
        # local backend is never selected automatically; users must opt in
        # by setting default_backend=local in config.
        if self.config.default_backend == "local":
            if request.enable_gpu:
                logger.debug("GPU requested, forcing docker backend instead of local")
                return "docker"
            if request.network_policy in ("whitelist", "proxy"):
                logger.debug(
                    "Restricted network policy requested, forcing docker backend instead of local"
                )
                return "docker"
            return "local"

        # Step 1: Check for secure requirements
        # Note: GPU and network isolation require firecracker, but we only have docker now
        # Fall back to docker until firecracker is implemented
        if request.enable_gpu:
            logger.debug("GPU requested, using docker backend (firecracker TBD)")
            return "docker"

        # Step 2: Check network policy
        if request.network_policy in ("whitelist", "proxy"):
            logger.debug(f"Network policy '{request.network_policy}' using docker backend (firecracker TBD)")
            return "docker"

        # Step 3: Route based on execution mode
        mode = request.mode.lower()
        if mode in self.MODE_BACKEND_MAP:
            mapped_backend = self.MODE_BACKEND_MAP[mode]
            if mapped_backend != "auto":
                return mapped_backend

        # Step 4: Route based on code scan result if provided
        if code_scan_result is not None:
            if code_scan_result.risk_score >= self.HIGH_RISK_THRESHOLD:
                logger.debug(
                    f"High risk code (score={code_scan_result.risk_score}), "
                    "using docker backend (firecracker TBD)"
                )
                return "docker"
            elif code_scan_result.risk_score >= self.MEDIUM_RISK_THRESHOLD:
                logger.debug(
                    f"Medium risk code (score={code_scan_result.risk_score}), "
                    "using docker backend"
                )
                return "docker"

        # Step 5: Use default backend from config
        if self.config.default_backend == "auto":
            return "docker"

        return self.config.default_backend

    def _validate_request(self, request: ExecutionRequest) -> None:
        """
        Validate execution request parameters.

        Args:
            request: Execution request to validate

        Raises:
            InvalidRequestError: If request is invalid
        """
        # Validate code is not empty
        if not request.code or not request.code.strip():
            raise InvalidRequestError(
                field="code",
                value=None,
                reason="Code cannot be empty",
            )

        # Validate workspace_id
        if not request.workspace_id:
            raise InvalidRequestError(
                field="workspace_id",
                value=None,
                reason="Workspace ID is required",
            )

        # Validate workspace_id format
        if len(request.workspace_id) < 1 or len(request.workspace_id) > 64:
            raise InvalidRequestError(
                field="workspace_id",
                value=request.workspace_id,
                reason="Workspace ID must be between 1 and 64 characters",
            )

        # Validate timeout
        if request.timeout_sec < 1 or request.timeout_sec > 86400:
            raise InvalidRequestError(
                field="timeout_sec",
                value=request.timeout_sec,
                reason="Timeout must be between 1 and 86400 seconds",
            )

        # Validate memory
        if request.memory_mb < 512 or request.memory_mb > 65536:
            raise InvalidRequestError(
                field="memory_mb",
                value=request.memory_mb,
                reason="Memory must be between 512 and 65536 MB",
            )

        # Validate CPU cores
        if request.cpu_cores < 0.5 or request.cpu_cores > 16.0:
            raise InvalidRequestError(
                field="cpu_cores",
                value=request.cpu_cores,
                reason="CPU cores must be between 0.5 and 16.0",
            )

        # Validate network whitelist if policy is whitelist
        if request.network_policy == "whitelist" and not request.network_whitelist:
            logger.warning(
                "Network policy is 'whitelist' but no whitelist entries provided. "
                "Network access will be blocked."
            )

    def get_supported_backends(self) -> List[str]:
        """
        Get list of supported backend names.

        Returns:
            List of backend name strings
        """
        backends = list(dict.fromkeys(self.MODE_BACKEND_MAP.values()))

        if "local" not in backends:
            backends.append("local")

        if self.config.default_backend != "auto" and self.config.default_backend not in backends:
            backends.append(self.config.default_backend)

        return backends
