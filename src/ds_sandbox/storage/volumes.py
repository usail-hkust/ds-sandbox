"""
Storage abstraction layer for ds-sandbox

Provides a flexible storage backend system with:
- StorageBackend: Abstract base class for storage implementations
- LocalVolumeBackend: Local filesystem storage with bind mount support
- Volume: Represents a storage volume with lifecycle management

Volume lifecycle: Created → Mounted → In Use → Unmounted
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import shutil
import uuid

try:
    from ..errors import SandboxError
except ImportError:
    # For standalone testing
    class SandboxError(Exception):
        def __init__(self, message, error_code=None, details=None):
            super().__init__(message)
            self.message = message
            self.error_code = error_code or "UNKNOWN"
            self.details = details or {}

logger = logging.getLogger(__name__)


class VolumeStatus(Enum):
    """Volume lifecycle status"""
    CREATED = "created"
    MOUNTED = "mounted"
    IN_USE = "in_use"
    UNMOUNTED = "unmounted"


class StorageType(Enum):
    """Supported storage backend types"""
    LOCAL = "local"
    NFS = "nfs"
    S3 = "s3"
    CEPH = "ceph"


class StorageError(SandboxError):
    """Base exception for storage operations"""

    def __init__(self, message: str, volume_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="STORAGE_ERROR", details=kwargs)
        self.volume_id = volume_id


class VolumeNotFoundError(StorageError):
    """Volume does not exist"""

    def __init__(self, volume_id: str):
        super().__init__(
            message=f"Volume '{volume_id}' not found",
            volume_id=volume_id
        )
        self.volume_id = volume_id


class VolumeMountError(StorageError):
    """Failed to mount volume"""

    def __init__(self, volume_id: str, reason: str):
        super().__init__(
            message=f"Failed to mount volume '{volume_id}': {reason}",
            volume_id=volume_id,
            reason=reason
        )


class VolumeUnmountError(StorageError):
    """Failed to unmount volume"""

    def __init__(self, volume_id: str, reason: str):
        super().__init__(
            message=f"Failed to unmount volume '{volume_id}': {reason}",
            volume_id=volume_id,
            reason=reason
        )


class VolumeInUseError(StorageError):
    """Volume is currently in use"""

    def __init__(self, volume_id: str):
        super().__init__(
            message=f"Volume '{volume_id}' is currently in use",
            volume_id=volume_id
        )


@dataclass
class Volume:
    """
    Represents a storage volume with lifecycle management.

    Attributes:
        volume_id: Unique identifier for the volume
        host_path: Path on the host machine
        guest_path: Path inside the sandbox
        mount_type: Type of mount (bind, tmpfs, etc.)
        status: Current lifecycle status
        size_bytes: Size of the volume in bytes (0 if unknown)
        created_at: ISO format timestamp
        mounted_at: ISO format timestamp when mounted
        options: Mount options (read-only, noexec, etc.)
        workspace_id: Associated workspace ID (if applicable)
        metadata: Additional metadata
    """

    volume_id: str
    host_path: str
    guest_path: str
    mount_type: str = "bind"
    status: VolumeStatus = VolumeStatus.CREATED
    size_bytes: int = 0
    created_at: str = field(default_factory=lambda: "")
    mounted_at: Optional[str] = None
    options: Dict[str, str] = field(default_factory=dict)
    workspace_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            from datetime import datetime
            self.created_at = datetime.utcnow().isoformat() + "Z"

    @property
    def is_mounted(self) -> bool:
        """Check if volume is mounted"""
        return self.status in (VolumeStatus.MOUNTED, VolumeStatus.IN_USE)

    @property
    def is_in_use(self) -> bool:
        """Check if volume is actively in use"""
        return self.status == VolumeStatus.IN_USE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "volume_id": self.volume_id,
            "host_path": self.host_path,
            "guest_path": self.guest_path,
            "mount_type": self.mount_type,
            "status": self.status.value,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "mounted_at": self.mounted_at,
            "options": self.options,
            "workspace_id": self.workspace_id,
            "metadata": self.metadata,
        }


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    Implementations must define:
    - create(): Create a new volume
    - mount(): Mount a volume for use
    - unmount(): Unmount a volume
    - delete(): Delete a volume
    - list(): List all volumes
    """

    def __init__(
        self,
        name: str,
        base_path: str,
        storage_type: StorageType = StorageType.LOCAL,
        **kwargs
    ):
        """
        Initialize storage backend.

        Args:
            name: Backend name identifier
            base_path: Base path for volumes
            storage_type: Type of storage backend
            **kwargs: Additional backend-specific options
        """
        self.name = name
        self.base_path = Path(base_path)
        self.storage_type = storage_type
        self._volumes: Dict[str, Volume] = {}
        self._mounted_volumes: Dict[str, Path] = {}  # volume_id -> mount_point
        self._initialized = False
        logger.info(f"Storage backend '{name}' initialized (type: {storage_type.value})")

    @abstractmethod
    def create_volume(
        self,
        volume_id: Optional[str] = None,
        size_bytes: int = 0,
        options: Optional[Dict[str, str]] = None,
        workspace_id: Optional[str] = None,
        **kwargs
    ) -> Volume:
        """
        Create a new storage volume.

        Args:
            volume_id: Optional custom volume ID (generated if not provided)
            size_bytes: Requested size in bytes (0 for auto-size)
            options: Mount options
            workspace_id: Associated workspace ID
            **kwargs: Backend-specific options

        Returns:
            Created Volume object
        """
        pass

    @abstractmethod
    def mount(self, volume: Volume, guest_path: str) -> str:
        """
        Mount a volume to a guest path.

        Args:
            volume: Volume to mount
            guest_path: Target path inside sandbox

        Returns:
            Actual guest path where mounted
        """
        pass

    @abstractmethod
    def unmount(self, volume: Volume) -> bool:
        """
        Unmount a volume.

        Args:
            volume: Volume to unmount

        Returns:
            True if successfully unmounted
        """
        pass

    @abstractmethod
    def delete(self, volume_id: str) -> bool:
        """
        Delete a volume.

        Args:
            volume_id: Volume to delete

        Returns:
            True if successfully deleted
        """
        pass

    @abstractmethod
    def list(self, workspace_id: Optional[str] = None) -> List[Volume]:
        """
        List volumes, optionally filtered by workspace.

        Args:
            workspace_id: Optional workspace filter

        Returns:
            List of Volume objects
        """
        pass

    @abstractmethod
    def get(self, volume_id: str) -> Optional[Volume]:
        """
        Get a volume by ID.

        Args:
            volume_id: Volume identifier

        Returns:
            Volume object or None
        """
        pass

    def initialize(self) -> None:
        """
        Initialize the storage backend.
        Called once during startup to set up necessary directories.
        """
        if self._initialized:
            return

        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            logger.info(f"Storage backend '{self.name}' initialized at {self.base_path}")
        except PermissionError as e:
            raise StorageError(
                f"Permission denied initializing storage at {self.base_path}: {e}",
                details={"base_path": str(self.base_path)}
            )
        except OSError as e:
            raise StorageError(
                f"Failed to initialize storage at {self.base_path}: {e}",
                details={"base_path": str(self.base_path)}
            )

    def cleanup(self) -> None:
        """
        Cleanup the storage backend.
        Called during shutdown to unmount and cleanup.
        """
        logger.info(f"Cleaning up storage backend '{self.name}'")
        for volume_id in list(self._volumes.keys()):
            try:
                volume = self._volumes[volume_id]
                if volume.is_mounted:
                    self.unmount(volume)
                self.delete(volume_id)
            except StorageError as e:
                logger.warning(f"Failed to cleanup volume {volume_id}: {e}")

    def register_volume(self, volume: Volume) -> None:
        """Register a volume in the backend"""
        self._volumes[volume.volume_id] = volume
        logger.debug(f"Registered volume {volume.volume_id}")

    def unregister_volume(self, volume_id: str) -> Optional[Volume]:
        """Unregister a volume from the backend"""
        return self._volumes.pop(volume_id, None)


class LocalVolumeBackend(StorageBackend):
    """
    Local filesystem storage backend using bind mounts.

    Provides fast, local storage access with:
    - Bind mount for host path access
    - Automatic directory creation
    - Workspace subdirectory support
    """

    def __init__(
        self,
        name: str = "local",
        base_path: str = "/opt/volumes",
        **kwargs
    ):
        """
        Initialize local volume backend.

        Args:
            name: Backend name
            base_path: Base directory for volumes
            **kwargs: Additional options (e.g., workspace_base for subdirectory creation)
        """
        super().__init__(name, base_path, StorageType.LOCAL, **kwargs)
        self.workspace_base = kwargs.get("workspace_base", "/opt/workspaces")

    def create_volume(
        self,
        volume_id: Optional[str] = None,
        size_bytes: int = 0,
        options: Optional[Dict[str, str]] = None,
        workspace_id: Optional[str] = None,
        host_path: Optional[str] = None,
        **kwargs
    ) -> Volume:
        """
        Create a new local volume.

        If host_path is provided, uses it directly. Otherwise creates
        a new volume directory under base_path.

        Args:
            volume_id: Optional custom volume ID
            size_bytes: Size hint (used for logging, not enforced)
            options: Mount options (read-only, noexec, etc.)
            workspace_id: Associated workspace ID
            host_path: Optional explicit host path (for bind mounts)

        Returns:
            Created Volume object
        """
        options = options or {}
        volume_id = volume_id or str(uuid.uuid4())

        # Determine host path
        if host_path:
            host_path_obj = Path(host_path)
        elif workspace_id:
            # Create in workspace subdirectory
            host_path_obj = Path(self.workspace_base) / workspace_id / "data" / volume_id
        else:
            host_path_obj = self.base_path / volume_id

        host_path_str = str(host_path_obj)

        # Create directory structure
        try:
            host_path_obj.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created volume directory: {host_path_str}")
        except PermissionError as e:
            raise StorageError(
                f"Permission denied creating volume at {host_path_str}: {e}",
                volume_id=volume_id
            )
        except OSError as e:
            raise StorageError(
                f"Failed to create volume at {host_path_str}: {e}",
                volume_id=volume_id
            )

        # Calculate actual size
        size_bytes = self._get_directory_size(host_path_obj)

        volume = Volume(
            volume_id=volume_id,
            host_path=host_path_str,
            guest_path="",  # Set during mount
            mount_type="bind",
            status=VolumeStatus.CREATED,
            size_bytes=size_bytes,
            options=options,
            workspace_id=workspace_id,
            metadata={"backend": self.name},
        )

        self.register_volume(volume)
        logger.info(f"Created volume {volume_id} at {host_path_str}")
        return volume

    def mount(self, volume: Volume, guest_path: str) -> str:
        """
        Mount a local volume using bind mount.

        For local storage, we use the same path on host and guest
        (bind mount semantics).

        Args:
            volume: Volume to mount
            guest_path: Target path in sandbox

        Returns:
            The guest path where volume is mounted
        """
        if not volume.host_path:
            raise VolumeMountError(
                volume.volume_id,
                "Volume has no host path configured"
            )

        # Verify host path exists
        if not Path(volume.host_path).exists():
            raise VolumeMountError(
                volume.volume_id,
                f"Host path does not exist: {volume.host_path}"
            )

        # Set guest path
        guest_path = guest_path or f"/mnt/volumes/{volume.volume_id}"
        volume.guest_path = guest_path
        volume.status = VolumeStatus.MOUNTED

        # For local bind mount, guest_path maps directly to host_path
        self._mounted_volumes[volume.volume_id] = Path(guest_path)

        logger.info(f"Mounted volume {volume.volume_id} at {guest_path}")
        return guest_path

    def unmount(self, volume: Volume) -> bool:
        """
        Unmount a local volume.

        Args:
            volume: Volume to unmount

        Returns:
            True if successfully unmounted
        """
        if volume.volume_id in self._mounted_volumes:
            del self._mounted_volumes[volume.volume_id]

        previous_status = volume.status
        volume.status = VolumeStatus.UNMOUNTED
        volume.mounted_at = None

        logger.info(f"Unmounted volume {volume.volume_id} (was: {previous_status.value})")
        return True

    def delete(self, volume_id: str) -> bool:
        """
        Delete a local volume.

        Args:
            volume_id: Volume to delete

        Returns:
            True if successfully deleted
        """
        volume = self.get(volume_id)
        if not volume:
            raise VolumeNotFoundError(volume_id)

        if volume.is_in_use:
            raise VolumeInUseError(volume_id)

        # Remove from mounted tracking
        if volume_id in self._mounted_volumes:
            del self._mounted_volumes[volume_id]

        # Delete directory if it exists and is within our base
        host_path = Path(volume.host_path)
        try:
            if host_path.exists():
                # Only delete if within our managed directories
                if str(host_path).startswith(str(self.base_path)):
                    shutil.rmtree(host_path)
                    logger.info(f"Deleted volume directory: {volume.host_path}")
                else:
                    logger.warning(
                        f"Skipping deletion of external path: {volume.host_path}"
                    )
        except PermissionError as e:
            raise StorageError(
                f"Permission denied deleting volume at {volume.host_path}: {e}",
                volume_id=volume_id
            )
        except OSError as e:
            raise StorageError(
                f"Failed to delete volume at {volume.host_path}: {e}",
                volume_id=volume_id
            )

        self.unregister_volume(volume_id)
        return True

    def list(self, workspace_id: Optional[str] = None) -> List[Volume]:
        """
        List local volumes, optionally filtered by workspace.

        Args:
            workspace_id: Optional workspace filter

        Returns:
            List of Volume objects
        """
        volumes = list(self._volumes.values())

        if workspace_id:
            volumes = [v for v in volumes if v.workspace_id == workspace_id]

        return volumes

    def get(self, volume_id: str) -> Optional[Volume]:
        """
        Get a volume by ID.

        Args:
            volume_id: Volume identifier

        Returns:
            Volume object or None
        """
        return self._volumes.get(volume_id)

    def create_workspace_subdirs(
        self,
        workspace_id: str,
        subdirs: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Create persistent subdirectories within a workspace.

        Args:
            workspace_id: Workspace ID
            subdirs: List of subdirectory names to create

        Returns:
            Dictionary mapping subdir names to their full paths
        """
        if subdirs is None:
            subdirs = ["data", "models", "outputs"]

        workspace_path = Path(self.workspace_base) / workspace_id
        created_paths: Dict[str, str] = {}

        for subdir in subdirs:
            subdir_path = workspace_path / subdir
            try:
                subdir_path.mkdir(parents=True, exist_ok=True)
                created_paths[subdir] = str(subdir_path)
                logger.info(f"Created workspace subdirectory: {subdir_path}")
            except PermissionError as e:
                raise StorageError(
                    f"Permission denied creating workspace subdirectory {subdir_path}: {e}",
                    volume_id=workspace_id
                )
            except OSError as e:
                raise StorageError(
                    f"Failed to create workspace subdirectory {subdir_path}: {e}",
                    volume_id=workspace_id
                )

        return created_paths

    def get_workspace_path(self, workspace_id: str, subdir: str = "") -> Path:
        """
        Get the full path for a workspace subdirectory.

        Args:
            workspace_id: Workspace ID
            subdir: Subdirectory name (empty for workspace root)

        Returns:
            Path object
        """
        workspace_path = Path(self.workspace_base) / workspace_id
        if subdir:
            workspace_path = workspace_path / subdir
        return workspace_path

    def _get_directory_size(self, path: Path) -> int:
        """Calculate total size of a directory in bytes"""
        total_size = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total_size += entry.stat().st_size
        except PermissionError:
            pass
        return total_size


# Factory function for creating storage backends
def create_storage_backend(
    backend_type: str = "local",
    name: str = "default",
    base_path: str = "/opt/volumes",
    **kwargs
) -> StorageBackend:
    """
    Create a storage backend by type.

    Args:
        backend_type: Type of backend (local, nfs, s3, ceph)
        name: Backend name
        base_path: Base path for volumes
        **kwargs: Additional backend-specific options

    Returns:
        StorageBackend instance
    """
    if backend_type == "local":
        return LocalVolumeBackend(name=name, base_path=base_path, **kwargs)
    else:
        raise ValueError(f"Unknown storage backend type: {backend_type}")


__all__ = [
    "StorageBackend",
    "LocalVolumeBackend",
    "Volume",
    "VolumeStatus",
    "StorageType",
    "StorageError",
    "VolumeNotFoundError",
    "VolumeMountError",
    "VolumeUnmountError",
    "VolumeInUseError",
    "create_storage_backend",
]
