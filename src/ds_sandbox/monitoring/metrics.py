"""
In-memory metrics collector for ds-sandbox.

This module keeps lightweight runtime metrics without external dependencies.
"""

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MetricsSnapshot:
    """Immutable view of current sandbox metrics."""

    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    stopped_executions: int = 0
    running_executions: int = 0
    active_workspaces: int = 0
    avg_execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return the snapshot as a plain dictionary."""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "stopped_executions": self.stopped_executions,
            "running_executions": self.running_executions,
            "active_workspaces": self.active_workspaces,
            "avg_execution_time_ms": self.avg_execution_time_ms,
        }


class InMemoryMetricsCollector:
    """
    Simple in-memory metrics collector.

    Tracks execution lifecycle and workspace activity.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._executions: Dict[str, Dict[str, Any]] = {}
        self._active_workspaces = set()

    def record_workspace_created(self, workspace_id: str) -> None:
        """Record a workspace creation event."""
        with self._lock:
            self._active_workspaces.add(workspace_id)

    def record_workspace_deleted(self, workspace_id: str) -> None:
        """Record a workspace deletion event."""
        with self._lock:
            self._active_workspaces.discard(workspace_id)

    def record_execution_started(self, execution_id: str, workspace_id: str) -> None:
        """Record the start of an execution."""
        with self._lock:
            self._executions[execution_id] = {
                "workspace_id": workspace_id,
                "status": "running",
                "duration_ms": None,
            }
            self._active_workspaces.add(workspace_id)

    def record_execution_completed(
        self,
        execution_id: str,
        status: str,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Record the completion of an execution."""
        with self._lock:
            if execution_id not in self._executions:
                self._executions[execution_id] = {
                    "workspace_id": None,
                    "status": status,
                    "duration_ms": duration_ms,
                }
                return

            self._executions[execution_id]["status"] = status
            if duration_ms is not None:
                self._executions[execution_id]["duration_ms"] = duration_ms

    def snapshot(self) -> Dict[str, Any]:
        """Compute and return current metrics snapshot."""
        with self._lock:
            executions = list(self._executions.values())
            active_workspaces = len(self._active_workspaces)

        total_executions = len(executions)
        successful_executions = sum(1 for e in executions if e.get("status") == "completed")
        failed_executions = sum(1 for e in executions if e.get("status") == "failed")
        stopped_executions = sum(1 for e in executions if e.get("status") == "stopped")
        running_executions = sum(1 for e in executions if e.get("status") == "running")

        duration_ms_list = [
            int(e["duration_ms"])
            for e in executions
            if e.get("status") in ("completed", "failed", "stopped")
            and e.get("duration_ms") is not None
        ]
        avg_execution_time_ms = (
            sum(duration_ms_list) / len(duration_ms_list) if duration_ms_list else 0.0
        )

        snapshot = MetricsSnapshot(
            total_executions=total_executions,
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            stopped_executions=stopped_executions,
            running_executions=running_executions,
            active_workspaces=active_workspaces,
            avg_execution_time_ms=round(avg_execution_time_ms, 2),
        )
        return snapshot.to_dict()

    def reset(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._executions.clear()
            self._active_workspaces.clear()
