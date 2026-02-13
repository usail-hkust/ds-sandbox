"""
Path validation utilities for ds-sandbox.

Prevents path traversal attacks by validating that user-supplied
path components do not escape their intended base directory.
"""

import os
from pathlib import Path
from typing import Union

from ds_sandbox.errors import InvalidRequestError


def validate_path_component(value: str, field_name: str = "path") -> None:
    """
    Validate that a user-supplied path component is safe to join
    with a base directory.

    Rules:
    - Must not be empty
    - Must not be an absolute path
    - Must not contain '..' segments
    - After normalization, must not escape the base (i.e. no leading '..')

    Args:
        value: The user-supplied path component (e.g. workspace_id,
               directory name, dataset name).
        field_name: Human-readable field name for error messages.

    Raises:
        InvalidRequestError: If the path component is unsafe.
    """
    if not value or not value.strip():
        raise InvalidRequestError(
            field=field_name,
            value=value,
            reason=f"{field_name} cannot be empty",
        )

    # Reject absolute paths (Unix and Windows)
    if os.path.isabs(value) or value.startswith("/") or value.startswith("\\"):
        raise InvalidRequestError(
            field=field_name,
            value=value,
            reason="Absolute paths are not allowed",
        )

    # Normalize and check for '..' traversal
    normalized = os.path.normpath(value)
    parts = normalized.split(os.sep)
    if ".." in parts:
        raise InvalidRequestError(
            field=field_name,
            value=value,
            reason="Path traversal ('..') is not allowed",
        )

    # Extra safety: on Windows normpath might produce a drive letter
    if os.path.isabs(normalized):
        raise InvalidRequestError(
            field=field_name,
            value=value,
            reason="Absolute paths are not allowed",
        )


def validate_resolved_path(child: Union[str, Path], base_dir: Union[str, Path]) -> Path:
    """
    Validate that a resolved child path stays within base_dir.

    This is a belt-and-suspenders check: even after component-level
    validation, we verify the resolved absolute path is under the
    expected base directory.

    Args:
        child: The resolved child path.
        base_dir: The base directory that the child must stay within.

    Returns:
        The resolved child Path.

    Raises:
        InvalidRequestError: If the resolved path escapes base_dir.
    """
    child_resolved = Path(child).resolve()
    base_resolved = Path(base_dir).resolve()

    # Use os.path.commonpath for a reliable prefix check
    try:
        common = Path(os.path.commonpath([str(child_resolved), str(base_resolved)]))
    except ValueError:
        # On Windows, paths on different drives raise ValueError
        raise InvalidRequestError(
            field="path",
            value=str(child),
            reason="Path is outside the allowed base directory",
        )

    if common != base_resolved:
        raise InvalidRequestError(
            field="path",
            value=str(child),
            reason="Path is outside the allowed base directory",
        )

    return child_resolved
