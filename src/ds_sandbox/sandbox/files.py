"""
File operations handler (E2B-compatible).
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class Files:
    """File operations handler (E2B-compatible)."""

    def __init__(self, sandbox: "Sandbox"):
        """Initialize with parent sandbox."""
        self._sandbox = sandbox

    def write(
        self,
        path: Union[str, List[Dict[str, Any]]],
        content: Optional[Union[bytes, str]] = None
    ) -> Union[str, List[str]]:
        """Write content to a file(s) synchronously (E2B-compatible).

        Args:
            path: Destination path in the sandbox, or a list of file dicts.
                  When a list is provided, each dict should have 'path' and 'data' keys.
                  Examples:
                    - "/home/user/data.csv" -> {workspace}/home/user/data.csv
                    - "data.csv" -> {workspace}/data.csv
                    - [{'path': '/a.txt', 'data': 'content'}, ...] -> multiple files
            content: File content as bytes or string (only used when path is a string)

        Returns:
            The path where the file was saved, or a list of created paths
        """
        return asyncio.run(self._write_async(path, content))

    def create(self, path: str, content: Union[bytes, str]) -> str:
        """Create a file with content (E2B-compatible alias for write).

        Args:
            path: Destination path in the sandbox.
            content: File content as bytes or string

        Returns:
            The path where the file was created
        """
        return self.write(path, content)

    async def _write_async(
        self,
        path: Union[str, List[Dict[str, Any]]],
        content: Optional[Union[bytes, str]] = None
    ) -> Union[str, List[str]]:
        """Write content to a file(s) asynchronously.

        Args:
            path: Destination path in the sandbox, or a list of file dicts.
                  When a list is provided, each dict should have 'path' and 'data' keys.
            content: File content as bytes or string (only used when path is a string)

        Returns:
            The path where the file was saved, or a list of created paths
        """
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        # Handle multiple files (array format)
        if isinstance(path, list):
            created_paths = []
            for file_spec in path:
                if not isinstance(file_spec, dict) or 'path' not in file_spec or 'data' not in file_spec:
                    raise ValueError("Each file spec must be a dict with 'path' and 'data' keys")
                file_path = file_spec['path']
                file_data = file_spec['data']
                rel_path = file_path.lstrip("/")
                host_path = Path(workspace.host_path) / rel_path
                host_path.parent.mkdir(parents=True, exist_ok=True)

                if isinstance(file_data, str):
                    host_path.write_text(file_data, encoding="utf-8")
                else:
                    host_path.write_bytes(file_data)

                created_paths.append(file_path)
                logger.debug(f"Wrote file to {host_path}")
            return created_paths

        # Handle single file (original behavior)
        if content is None:
            raise ValueError("content parameter is required when path is a string")

        rel_path = path.lstrip("/")
        host_path = Path(workspace.host_path) / rel_path
        host_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, str):
            host_path.write_text(content, encoding="utf-8")
        else:
            host_path.write_bytes(content)

        logger.debug(f"Wrote file to {host_path}")
        return path

    def read(self, path: str) -> bytes:
        """Read content from a file synchronously (E2B-compatible).

        Args:
            path: Path to the file in the sandbox.

        Returns:
            File content as bytes
        """
        return asyncio.run(self._read_async(path))

    async def _read_async(self, path: str) -> bytes:
        """Read content from a file in the sandbox asynchronously.

        Args:
            path: Path to the file in the sandbox.

        Returns:
            File content as bytes
        """
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        # 直接拼接到 workspace 目录
        rel_path = path.lstrip("/")
        host_path = Path(workspace.host_path) / rel_path

        if not host_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return host_path.read_bytes()

    def list(self, path: str = None) -> List[Dict[str, Any]]:
        """List files in a directory synchronously (E2B-compatible).

        Args:
            path: Directory path in the sandbox (default: workspace root)

        Returns:
            List of file/directory information
        """
        return asyncio.run(self._list_async(path))

    async def _list_async(self, path: str = None) -> List[Dict[str, Any]]:
        """List files in a directory.

        Args:
            path: Directory path in the sandbox (default: workspace root)

        Returns:
            List of file/directory information
        """
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        # 默认列出 workspace 根目录
        if path is None:
            host_path = Path(workspace.host_path)
        else:
            rel_path = path.lstrip("/")
            host_path = Path(workspace.host_path) / rel_path

        if not host_path.exists():
            return []

        results = []
        for item in host_path.iterdir():
            results.append({
                "name": item.name,
                "path": str(item),
                "is_file": item.is_file(),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else 0,
            })

        return results

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists (E2B-compatible).

        Args:
            path: Path to check in the sandbox.

        Returns:
            True if file or directory exists
        """
        return asyncio.run(self._exists_async(path))

    async def _exists_async(self, path: str) -> bool:
        """Check if a file or directory exists asynchronously."""
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        rel_path = path.lstrip("/")
        host_path = Path(workspace.host_path) / rel_path
        return host_path.exists()

    def get_info(self, path: str) -> Dict[str, Any]:
        """Get information about a file or directory (E2B-compatible).

        Args:
            path: Path to get info for in the sandbox.

        Returns:
            Dictionary with file/directory information
        """
        return asyncio.run(self._get_info_async(path))

    async def _get_info_async(self, path: str) -> Dict[str, Any]:
        """Get information about a file or directory asynchronously."""
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        rel_path = path.lstrip("/")
        host_path = Path(workspace.host_path) / rel_path

        if not host_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        stat = host_path.stat()
        return {
            "name": host_path.name,
            "path": str(host_path),
            "is_file": host_path.is_file(),
            "is_dir": host_path.is_dir(),
            "size": stat.st_size,
            "created_at": stat.st_ctime,
            "modified_at": stat.st_mtime,
        }

    def remove(self, path: str) -> None:
        """Remove a file or directory (E2B-compatible).

        Args:
            path: Path to remove in the sandbox.
        """
        return asyncio.run(self._remove_async(path))

    async def _remove_async(self, path: str) -> None:
        """Remove a file or directory asynchronously."""
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        rel_path = path.lstrip("/")
        host_path = Path(workspace.host_path) / rel_path

        if not host_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if host_path.is_file():
            host_path.unlink()
        elif host_path.is_dir():
            import shutil
            shutil.rmtree(host_path)

        logger.debug(f"Removed: {host_path}")

    def rename(self, old_path: str, new_path: str) -> str:
        """Rename a file or directory (E2B-compatible).

        Args:
            old_path: Current path in the sandbox.
            new_path: New path in the sandbox.

        Returns:
            The new path
        """
        return asyncio.run(self._rename_async(old_path, new_path))

    async def _rename_async(self, old_path: str, new_path: str) -> str:
        """Rename a file or directory asynchronously."""
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        old_rel_path = old_path.lstrip("/")
        new_rel_path = new_path.lstrip("/")

        old_host_path = Path(workspace.host_path) / old_rel_path
        new_host_path = Path(workspace.host_path) / new_rel_path

        if not old_host_path.exists():
            raise FileNotFoundError(f"Path not found: {old_path}")

        old_host_path.rename(new_host_path)
        logger.debug(f"Renamed: {old_host_path} -> {new_host_path}")
        return new_path

    def make_dir(self, path: str) -> str:
        """Create a directory (E2B-compatible).

        Args:
            path: Directory path to create in the sandbox.

        Returns:
            The created directory path
        """
        return asyncio.run(self._make_dir_async(path))

    async def _make_dir_async(self, path: str) -> str:
        """Create a directory asynchronously."""
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        rel_path = path.lstrip("/")
        host_path = Path(workspace.host_path) / rel_path

        host_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {host_path}")
        return path

    def write_files(self, files: Dict[str, Union[bytes, str]]) -> List[str]:
        """Write multiple files at once (E2B-compatible).

        Args:
            files: Dictionary mapping paths to content

        Returns:
            List of created file paths
        """
        return asyncio.run(self._write_files_async(files))

    async def _write_files_async(self, files: Dict[str, Union[bytes, str]]) -> List[str]:
        """Write multiple files at once asynchronously."""
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        created_paths = []
        for path, content in files.items():
            rel_path = path.lstrip("/")
            host_path = Path(workspace.host_path) / rel_path
            host_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(content, str):
                host_path.write_text(content, encoding="utf-8")
            else:
                host_path.write_bytes(content)

            created_paths.append(path)
            logger.debug(f"Wrote file: {host_path}")

        return created_paths

    def watch_dir(
        self,
        path: str,
        callback: Callable[[Dict[str, Any]], None],
        recursive: bool = False,
    ) -> None:
        """Watch a directory for changes synchronously (E2B-compatible).

        Args:
            path: Directory path to watch in the sandbox.
            callback: Callback function called on file changes.
            recursive: Whether to watch subdirectories recursively.
        """
        return asyncio.run(self._watch_dir_async(path, callback, recursive))

    async def _watch_dir_async(
        self,
        path: str,
        callback: Callable[[Dict[str, Any]], None],
        recursive: bool = False,
    ) -> None:
        """Watch a directory for changes (E2B-compatible).

        Args:
            path: Directory path to watch in the sandbox.
            callback: Callback function called on file changes.
            recursive: Whether to watch subdirectories recursively.
        """
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        rel_path = path.lstrip("/")
        host_path = Path(workspace.host_path) / rel_path

        if not host_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        # Use watchdog for directory monitoring
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class ChangeHandler(FileSystemEventHandler):
                def __init__(self, cb):
                    self._cb = cb

                def on_any_event(self, event):
                    if not event.is_directory:
                        self._cb({
                            "type": event.event_type,
                            "path": event.src_path,
                            "is_directory": event.is_directory,
                        })

            handler = ChangeHandler(callback)
            observer = Observer()
            observer.schedule(handler, str(host_path), recursive=recursive)
            observer.start()

            # Keep observer alive - caller should handle stopping
            self._watch_observer = observer
        except ImportError:
            logger.warning("watchdog not installed, watch_dir will not function")

    def upload(self, local_path: str, remote_path: str) -> str:
        """Upload a local file to the sandbox (E2B-compatible).

        Args:
            local_path: Path to local file
            remote_path: Destination path in sandbox

        Returns:
            The remote path where file was saved
        """
        return asyncio.run(self._upload_async(local_path, remote_path))

    async def _upload_async(self, local_path: str, remote_path: str) -> str:
        """Upload a local file to the sandbox asynchronously.

        Args:
            local_path: Path to local file
            remote_path: Destination path in sandbox

        Returns:
            The remote path where file was saved
        """
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        local_path_obj = Path(local_path)
        if not local_path_obj.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        if not local_path_obj.is_file():
            raise ValueError(f"Local path is not a file: {local_path}")

        rel_path = remote_path.lstrip("/")
        host_path = Path(workspace.host_path) / rel_path
        host_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file content
        import shutil
        shutil.copy2(local_path_obj, host_path)

        logger.debug(f"Uploaded file to {host_path}")
        return remote_path

    def download(self, remote_path: str, local_path: str) -> str:
        """Download a file from sandbox to local (E2B-compatible).

        Args:
            remote_path: Path to file in sandbox
            local_path: Destination local path

        Returns:
            The local path where file was saved
        """
        return asyncio.run(self._download_async(remote_path, local_path))

    async def _download_async(self, remote_path: str, local_path: str) -> str:
        """Download a file from sandbox to local asynchronously.

        Args:
            remote_path: Path to file in sandbox
            local_path: Destination local path

        Returns:
            The local path where file was saved
        """
        workspace = self._sandbox.workspace
        if not workspace:
            raise RuntimeError("Sandbox not initialized. Call await sandbox.create() first.")

        rel_path = remote_path.lstrip("/")
        host_path = Path(workspace.host_path) / rel_path

        if not host_path.exists():
            raise FileNotFoundError(f"File not found in sandbox: {remote_path}")

        if not host_path.is_file():
            raise ValueError(f"Path is not a file in sandbox: {remote_path}")

        local_path_obj = Path(local_path)
        local_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Copy file content
        import shutil
        shutil.copy2(host_path, local_path_obj)

        logger.debug(f"Downloaded file to {local_path}")
        return local_path


# Import Sandbox type hint (lazy import to avoid circular imports)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ds_sandbox.sandbox.sandbox import Sandbox
