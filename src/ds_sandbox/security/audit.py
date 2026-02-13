"""
ds-sandbox audit logging module

Provides comprehensive audit logging for sandbox operations:
- Execution events (start, complete, fail, timeout)
- Workspace events (create, delete)
- Data preparation events
- Security scan events

Supports multiple output formats:
- JSON file output
- Structured logging
- Extensible for remote log services
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events"""

    EXECUTION_STARTED = "EXECUTION_STARTED"
    EXECUTION_COMPLETED = "EXECUTION_COMPLETED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
    WORKSPACE_CREATED = "WORKSPACE_CREATED"
    WORKSPACE_DELETED = "WORKSPACE_DELETED"
    DATASET_PREPARED = "DATASET_PREPARED"
    SECURITY_SCAN_COMPLETED = "SECURITY_SCAN_COMPLETED"
    SECURITY_SCAN_REJECTED = "SECURITY_SCAN_REJECTED"


class AuditEntry(BaseModel):
    """Audit log entry model"""

    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp in UTC"
    )

    event_type: str = Field(
        ...,
        description="Type of audit event"
    )

    execution_id: Optional[str] = Field(
        None,
        description="Execution ID (if applicable)"
    )

    workspace_id: Optional[str] = Field(
        None,
        description="Workspace ID (if applicable)"
    )

    user_id: Optional[str] = Field(
        None,
        description="User identifier"
    )

    code_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the executed code"
    )

    backend: Optional[str] = Field(
        None,
        description="Backend used (docker/firecracker/kata)"
    )

    isolation_level: Optional[str] = Field(
        None,
        description="Isolation level applied"
    )

    risk_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Risk score (0.0 - 1.0)"
    )

    duration_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Duration in milliseconds"
    )

    success: Optional[bool] = Field(
        None,
        description="Whether the operation succeeded"
    )

    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event-specific details"
    )

    @classmethod
    def create(
        cls,
        event_type: AuditEventType,
        execution_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        code: Optional[str] = None,
        backend: Optional[str] = None,
        isolation_level: Optional[str] = None,
        risk_score: Optional[float] = None,
        duration_ms: Optional[int] = None,
        success: Optional[bool] = None,
        **details: Any,
    ) -> "AuditEntry":
        """
        Create an audit entry with common fields.

        Args:
            event_type: Type of the event
            execution_id: Execution ID
            workspace_id: Workspace ID
            user_id: User identifier
            code: Code string (will be hashed)
            backend: Backend used
            isolation_level: Isolation level
            risk_score: Risk score
            duration_ms: Duration in milliseconds
            success: Success status
            **details: Additional details

        Returns:
            AuditEntry instance
        """
        # Compute code hash if code is provided
        code_hash = None
        if code:
            code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()

        return cls(
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
            event_type=event_type.value,
            execution_id=execution_id,
            workspace_id=workspace_id,
            user_id=user_id,
            code_hash=code_hash,
            backend=backend,
            isolation_level=isolation_level,
            risk_score=risk_score,
            duration_ms=duration_ms,
            success=success,
            details=details,
        )


class AuditOutput(ABC):
    """Abstract base class for audit log outputs"""

    @abstractmethod
    def write(self, entry: AuditEntry) -> None:
        """Write an audit entry"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the output and release resources"""
        pass


class JSONFileOutput(AuditOutput):
    """JSON file output for audit logs"""

    def __init__(
        self,
        file_path: str,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        mode: str = "a",
    ):
        """
        Initialize JSON file output.

        Args:
            file_path: Path to the log file
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            mode: File open mode ('a' for append, 'w' for write)
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._init_handler(max_bytes, backup_count, mode)

    def _init_handler(
        self,
        max_bytes: int,
        backup_count: int,
        mode: str,
    ) -> None:
        """Initialize the file handler with rotation"""
        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        self._handler = RotatingFileHandler(
            str(self.file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            mode=mode,
        )
        self._handler.setLevel(logging.DEBUG)

        # Create JSON formatter
        formatter = logging.Formatter("%(message)s")
        self._handler.setFormatter(formatter)

        # Create dedicated logger for audit
        self._logger = logging.getLogger(f"audit.file.{self.file_path.stem}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(self._handler)
        self._logger.propagate = False

    def write(self, entry: AuditEntry) -> None:
        """Write an audit entry to the file"""
        json_line = entry.model_dump_json(exclude_none=True)
        self._logger.debug(json_line)

    def close(self) -> None:
        """Close the file handler"""
        if hasattr(self, "_handler"):
            self._handler.close()
            self._handler = None

    def __del__(self):
        self.close()


class StructuredLogOutput(AuditOutput):
    """Structured logging output using Python's logging module"""

    def __init__(
        self,
        logger_name: str = "audit",
        level: int = logging.INFO,
    ):
        """
        Initialize structured log output.

        Args:
            logger_name: Name of the logger to use
            level: Log level
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(timestamp)s %(levelname)s %(event_type)s "
                "%(execution_id)s %(workspace_id)s %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def write(self, entry: AuditEntry) -> None:
        """Write an audit entry as structured log"""
        extra = {
            "timestamp": entry.timestamp,
            "event_type": entry.event_type,
            "execution_id": entry.execution_id or "",
            "workspace_id": entry.workspace_id or "",
        }

        if entry.success is True:
            self.logger.info(
                f"Audit event: {entry.event_type}",
                extra=extra,
                stack_info=False,
                exc_info=False,
            )
        elif entry.success is False:
            self.logger.warning(
                f"Audit event failed: {entry.event_type}",
                extra=extra,
                stack_info=False,
                exc_info=False,
            )
        else:
            self.logger.debug(
                f"Audit event: {entry.event_type}",
                extra=extra,
                stack_info=False,
                exc_info=False,
            )

    def close(self) -> None:
        """No-op for structured log output"""
        pass


class AuditLogger:
    """
    Comprehensive audit logger for sandbox operations.

    Features:
    - Multiple output destinations (JSON file, structured log)
    - Asynchronous writes for performance
    - Log rotation
    - Event-type specific logging methods
    """

    def __init__(
        self,
        outputs: Optional[List[AuditOutput]] = None,
        enable_async: bool = True,
    ):
        """
        Initialize the audit logger.

        Args:
            outputs: List of output handlers (creates defaults if empty)
            enable_async: Whether to use async writes
        """
        self.outputs = outputs or []
        self.enable_async = enable_async
        self._write_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

        # Setup default outputs if none provided
        if not self.outputs:
            self._setup_default_outputs()

    def _setup_default_outputs(self) -> None:
        """Setup default output handlers"""
        # Add structured log output by default
        self.outputs.append(StructuredLogOutput())

    def start(self) -> None:
        """Start the async write worker"""
        if self.enable_async and not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._write_worker())

    def stop(self) -> None:
        """Stop the async write worker and flush pending writes"""
        self._running = False

        if self._worker_task:
            self._worker_task.cancel()
            self._worker_task = None

    def _write(self, entry: AuditEntry) -> None:
        """Synchronous write to all outputs"""
        for output in self.outputs:
            try:
                output.write(entry)
            except Exception as e:
                logger.error(f"Failed to write audit entry: {e}")

    async def _write_async(self, entry: AuditEntry) -> None:
        """Queue an audit entry for async writing"""
        await self._write_queue.put(entry)

    async def _write_worker(self) -> None:
        """Background worker that processes the write queue"""
        while self._running:
            try:
                entry = await asyncio.wait_for(
                    self._write_queue.get(),
                    timeout=1.0,
                )
                self._write(entry)
                self._write_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Audit write worker error: {e}")

    def log(self, entry: AuditEntry) -> None:
        """
        Log an audit entry.

        Args:
            entry: The audit entry to log
        """
        if self.enable_async:
            asyncio.create_task(self._write_async(entry))
        else:
            self._write(entry)

    # Convenience methods for specific event types

    def log_execution_started(
        self,
        execution_id: str,
        workspace_id: str,
        user_id: Optional[str] = None,
        code: Optional[str] = None,
        backend: Optional[str] = None,
        isolation_level: Optional[str] = None,
        timeout_sec: Optional[int] = None,
        memory_mb: Optional[int] = None,
        **details,
    ) -> None:
        """
        Log execution started event.

        Args:
            execution_id: Unique execution identifier
            workspace_id: Workspace identifier
            user_id: User identifier
            code: Code being executed
            backend: Backend being used
            isolation_level: Isolation level
            timeout_sec: Timeout in seconds
            memory_mb: Memory limit in MB
            **details: Additional details
        """
        entry = AuditEntry.create(
            event_type=AuditEventType.EXECUTION_STARTED,
            execution_id=execution_id,
            workspace_id=workspace_id,
            user_id=user_id,
            code=code,
            backend=backend,
            isolation_level=isolation_level,
            risk_score=0.0,  # Initial risk, will be updated on completion
            **details,
        )
        entry.details.update({
            "timeout_sec": timeout_sec,
            "memory_mb": memory_mb,
        })
        self.log(entry)

    def log_execution_completed(
        self,
        execution_id: str,
        workspace_id: str,
        user_id: Optional[str] = None,
        code: Optional[str] = None,
        backend: Optional[str] = None,
        isolation_level: Optional[str] = None,
        risk_score: Optional[float] = None,
        duration_ms: Optional[int] = None,
        success: bool = True,
        **details,
    ) -> None:
        """
        Log execution completed event.

        Args:
            execution_id: Unique execution identifier
            workspace_id: Workspace identifier
            user_id: User identifier
            code: Code that was executed
            backend: Backend that was used
            isolation_level: Isolation level applied
            risk_score: Final risk score
            duration_ms: Execution duration in milliseconds
            success: Whether execution succeeded
            **details: Additional details
        """
        entry = AuditEntry.create(
            event_type=AuditEventType.EXECUTION_COMPLETED,
            execution_id=execution_id,
            workspace_id=workspace_id,
            user_id=user_id,
            code=code,
            backend=backend,
            isolation_level=isolation_level,
            risk_score=risk_score,
            duration_ms=duration_ms,
            success=success,
            **details,
        )
        self.log(entry)

    def log_execution_failed(
        self,
        execution_id: str,
        workspace_id: str,
        user_id: Optional[str] = None,
        code: Optional[str] = None,
        backend: Optional[str] = None,
        duration_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        **details,
    ) -> None:
        """
        Log execution failed event.

        Args:
            execution_id: Unique execution identifier
            workspace_id: Workspace identifier
            user_id: User identifier
            code: Code that was executed
            backend: Backend that was used
            duration_ms: Execution duration in milliseconds
            error_message: Error message
            **details: Additional details
        """
        entry = AuditEntry.create(
            event_type=AuditEventType.EXECUTION_FAILED,
            execution_id=execution_id,
            workspace_id=workspace_id,
            user_id=user_id,
            code=code,
            backend=backend,
            risk_score=1.0,  # Maximum risk for failures
            duration_ms=duration_ms,
            success=False,
            **details,
        )
        entry.details.update({
            "error_message": error_message,
        })
        self.log(entry)

    def log_execution_timeout(
        self,
        execution_id: str,
        workspace_id: str,
        user_id: Optional[str] = None,
        code: Optional[str] = None,
        backend: Optional[str] = None,
        duration_ms: Optional[int] = None,
        timeout_sec: Optional[int] = None,
        **details,
    ) -> None:
        """
        Log execution timeout event.

        Args:
            execution_id: Unique execution identifier
            workspace_id: Workspace identifier
            user_id: User identifier
            code: Code that was executed
            backend: Backend that was used
            duration_ms: Execution duration in milliseconds
            timeout_sec: Timeout that was set
            **details: Additional details
        """
        entry = AuditEntry.create(
            event_type=AuditEventType.EXECUTION_TIMEOUT,
            execution_id=execution_id,
            workspace_id=workspace_id,
            user_id=user_id,
            code=code,
            backend=backend,
            risk_score=0.5,  # Medium risk for timeouts
            duration_ms=duration_ms,
            success=False,
            **details,
        )
        entry.details.update({
            "timeout_sec": timeout_sec,
        })
        self.log(entry)

    def log_workspace_created(
        self,
        workspace_id: str,
        user_id: Optional[str] = None,
        host_path: Optional[str] = None,
        **details,
    ) -> None:
        """
        Log workspace created event.

        Args:
            workspace_id: Workspace identifier
            user_id: User who created the workspace
            host_path: Host path for the workspace
            **details: Additional details
        """
        entry = AuditEntry.create(
            event_type=AuditEventType.WORKSPACE_CREATED,
            workspace_id=workspace_id,
            user_id=user_id,
            **details,
        )
        entry.details.update({
            "host_path": host_path,
        })
        self.log(entry)

    def log_workspace_deleted(
        self,
        workspace_id: str,
        user_id: Optional[str] = None,
        reason: Optional[str] = None,
        **details,
    ) -> None:
        """
        Log workspace deleted event.

        Args:
            workspace_id: Workspace identifier
            user_id: User who deleted the workspace
            reason: Reason for deletion
            **details: Additional details
        """
        entry = AuditEntry.create(
            event_type=AuditEventType.WORKSPACE_DELETED,
            workspace_id=workspace_id,
            user_id=user_id,
            success=True,
            **details,
        )
        entry.details.update({
            "reason": reason,
        })
        self.log(entry)

    def log_dataset_prepared(
        self,
        workspace_id: str,
        dataset_name: str,
        user_id: Optional[str] = None,
        size_mb: Optional[float] = None,
        strategy: Optional[str] = None,
        **details,
    ) -> None:
        """
        Log dataset prepared event.

        Args:
            workspace_id: Workspace identifier
            dataset_name: Name of the dataset
            user_id: User who prepared the dataset
            size_mb: Size of the dataset in MB
            strategy: Preparation strategy (copy/link)
            **details: Additional details
        """
        entry = AuditEntry.create(
            event_type=AuditEventType.DATASET_PREPARED,
            workspace_id=workspace_id,
            user_id=user_id,
            success=True,
            **details,
        )
        entry.details.update({
            "dataset_name": dataset_name,
            "size_mb": size_mb,
            "strategy": strategy,
        })
        self.log(entry)

    def log_security_scan_completed(
        self,
        execution_id: str,
        workspace_id: str,
        user_id: Optional[str] = None,
        code: Optional[str] = None,
        risk_score: Optional[float] = None,
        issues_count: Optional[int] = None,
        recommended_isolation: Optional[str] = None,
        **details,
    ) -> None:
        """
        Log security scan completed event.

        Args:
            execution_id: Execution identifier
            workspace_id: Workspace identifier
            user_id: User who submitted the code
            code: Code that was scanned
            risk_score: Calculated risk score
            issues_count: Number of issues found
            recommended_isolation: Recommended isolation level
            **details: Additional details
        """
        entry = AuditEntry.create(
            event_type=AuditEventType.SECURITY_SCAN_COMPLETED,
            execution_id=execution_id,
            workspace_id=workspace_id,
            user_id=user_id,
            code=code,
            risk_score=risk_score,
            success=True,
            **details,
        )
        entry.details.update({
            "issues_count": issues_count,
            "recommended_isolation": recommended_isolation,
        })
        self.log(entry)

    def log_security_scan_rejected(
        self,
        execution_id: str,
        workspace_id: str,
        user_id: Optional[str] = None,
        code: Optional[str] = None,
        risk_score: Optional[float] = None,
        rejection_reason: Optional[str] = None,
        **details,
    ) -> None:
        """
        Log security scan rejected event.

        Args:
            execution_id: Execution identifier
            workspace_id: Workspace identifier
            user_id: User who submitted the code
            code: Code that was rejected
            risk_score: Calculated risk score
            rejection_reason: Reason for rejection
            **details: Additional details
        """
        entry = AuditEntry.create(
            event_type=AuditEventType.SECURITY_SCAN_REJECTED,
            execution_id=execution_id,
            workspace_id=workspace_id,
            user_id=user_id,
            code=code,
            risk_score=risk_score,
            success=False,
            **details,
        )
        entry.details.update({
            "rejection_reason": rejection_reason,
        })
        self.log(entry)

    def close(self) -> None:
        """Close all output handlers"""
        for output in self.outputs:
            try:
                output.close()
            except Exception as e:
                logger.error(f"Error closing audit output: {e}")

    def __del__(self):
        self.close()


def create_audit_logger(
    log_file_path: Optional[str] = None,
    enable_structured_log: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    enable_async: bool = True,
) -> AuditLogger:
    """
    Factory function to create an audit logger with standard configuration.

    Args:
        log_file_path: Path to JSON log file (optional)
        enable_structured_log: Whether to enable structured logging
        max_bytes: Maximum file size for rotation
        backup_count: Number of backup files
        enable_async: Whether to use async writes

    Returns:
        Configured AuditLogger instance
    """
    outputs: List[AuditOutput] = []

    if log_file_path:
        outputs.append(JSONFileOutput(
            file_path=log_file_path,
            max_bytes=max_bytes,
            backup_count=backup_count,
        ))

    if enable_structured_log:
        outputs.append(StructuredLogOutput())

    logger = AuditLogger(outputs=outputs, enable_async=enable_async)
    logger.start()

    return logger
