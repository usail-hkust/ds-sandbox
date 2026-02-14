"""
Execution tracking functionality for the sandbox manager.

This module provides the ExecutionTracker class for tracking
execution state, metrics, and events.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ds_sandbox.monitoring.metrics import InMemoryMetricsCollector
from ds_sandbox.types import SandboxEvent, SandboxMetrics

logger = logging.getLogger(__name__)


class ExecutionTracker:
    """
    Execution tracker - manages execution state, metrics, and events.

    This class handles:
    - Tracking execution start/completion
    - Collecting and storing metrics
    - Emitting and retrieving events
    """

    def __init__(self):
        """Initialize the execution tracker."""
        self._execution_store: Dict[str, Dict[str, Any]] = {}
        self._metrics = InMemoryMetricsCollector()
        self._events: List[SandboxEvent] = []
        self._metrics_history: Dict[str, List[SandboxMetrics]] = {}

    def _track_execution_start(
        self,
        execution_id: str,
        workspace_id: str,
        backend: str,
    ) -> None:
        """Track the start of an execution."""
        self._execution_store[execution_id] = {
            "execution_id": execution_id,
            "workspace_id": workspace_id,
            "status": "running",
            "backend": backend,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        }
        self._metrics.record_execution_started(execution_id, workspace_id)

    def _track_execution_complete(
        self,
        execution_id: str,
        status: str,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        exit_code: Optional[int] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Track the completion of an execution."""
        if execution_id in self._execution_store:
            self._execution_store[execution_id]["status"] = status
            self._execution_store[execution_id]["completed_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            if stdout is not None:
                self._execution_store[execution_id]["stdout"] = stdout
            if stderr is not None:
                self._execution_store[execution_id]["stderr"] = stderr
            if exit_code is not None:
                self._execution_store[execution_id]["exit_code"] = exit_code
            if duration_ms is not None:
                self._execution_store[execution_id]["duration_ms"] = duration_ms
            self._metrics.record_execution_completed(
                execution_id,
                status,
                duration_ms=duration_ms,
            )

    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get the status of an execution.

        Args:
            execution_id: Execution identifier

        Returns:
            Dictionary with execution status
        """
        if execution_id not in self._execution_store:
            from ds_sandbox.errors import ExecutionNotFoundError
            raise ExecutionNotFoundError(execution_id=execution_id)
        return self._execution_store[execution_id]

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics.

        Returns:
            Dictionary with system metrics
        """
        return self._metrics.snapshot()

    def get_workspace_metrics(self, workspace_id: str) -> List[SandboxMetrics]:
        """
        Get metrics history for a workspace.

        Args:
            workspace_id: Workspace identifier

        Returns:
            List of SandboxMetrics objects
        """
        return self._metrics_history.get(workspace_id, [])

    def get_system_metrics(self) -> List[SandboxMetrics]:
        """
        Get all system metrics across all workspaces.

        Returns:
            List of all SandboxMetrics objects
        """
        all_metrics = []
        for metrics_list in self._metrics_history.values():
            all_metrics.extend(metrics_list)
        # Return sorted by timestamp
        all_metrics.sort(key=lambda m: m.timestamp)
        return all_metrics

    def collect_workspace_metrics(self, workspace_id: str) -> SandboxMetrics:
        """
        Collect current metrics for a workspace.

        Args:
            workspace_id: Workspace identifier

        Returns:
            SandboxMetrics object with current metrics
        """
        import psutil

        # Get CPU count
        cpu_count = psutil.cpu_count() or 1

        # Get CPU usage (percentage over 0.1 second interval)
        cpu_used_pct = psutil.cpu_percent(interval=0.1)

        # Get memory info
        mem = psutil.virtual_memory()
        mem_total_mib = int(mem.total / (1024 * 1024))
        mem_used_mib = int(mem.used / (1024 * 1024))

        # Get current timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create metrics object
        metrics = SandboxMetrics(
            cpu_count=cpu_count,
            cpu_used_pct=cpu_used_pct,
            mem_total_mib=mem_total_mib,
            mem_used_mib=mem_used_mib,
            timestamp=timestamp,
        )

        # Store in history
        if workspace_id not in self._metrics_history:
            self._metrics_history[workspace_id] = []

        # Keep only last 100 metrics per workspace
        if len(self._metrics_history[workspace_id]) >= 100:
            self._metrics_history[workspace_id] = self._metrics_history[workspace_id][-99:]

        self._metrics_history[workspace_id].append(metrics)

        return metrics

    # =========================================================================
    # Event Management
    # =========================================================================

    def emit_event(
        self,
        event_type: str,
        workspace_id: str,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> SandboxEvent:
        """
        Emit a sandbox lifecycle event.

        Args:
            event_type: Type of event (e.g., "sandbox.lifecycle.created")
            workspace_id: Workspace ID
            event_data: Additional event data

        Returns:
            The created SandboxEvent
        """
        event = SandboxEvent(
            id=f"evt-{uuid.uuid4().hex[:12]}",
            type=event_type,
            event_data=event_data or {},
            sandbox_id=f"sandbox-{workspace_id}",
            workspace_id=workspace_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._events.append(event)
        logger.info(f"Event emitted: {event_type} for workspace {workspace_id}")
        return event

    def get_events(
        self,
        workspace_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[SandboxEvent]:
        """
        Get sandbox lifecycle events.

        Args:
            workspace_id: Optional workspace ID to filter events
            limit: Maximum number of events to return (default: 10)

        Returns:
            List of SandboxEvent objects
        """
        events = self._events

        if workspace_id:
            events = [e for e in events if e.workspace_id == workspace_id]

        # Return most recent events (last N)
        return events[-limit:] if len(events) > limit else events

    def get_all_events(self, limit: int = 100) -> List[SandboxEvent]:
        """
        Get all sandbox lifecycle events (admin endpoint).

        Args:
            limit: Maximum number of events to return (default: 100)

        Returns:
            List of all SandboxEvent objects
        """
        # Return most recent events (last N)
        return self._events[-limit:] if len(self._events) > limit else self._events
