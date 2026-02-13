"""
Dataset Registry

Centralized dataset management for ds-sandbox.
Handles dataset registration, storage, and retrieval.
"""

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..errors import DatasetNotFoundError
from ..types import DatasetInfo

logger = logging.getLogger(__name__)

INDEX_FILENAME = ".index.json"


class DatasetRegistry:
    """
    Central registry for managing datasets.

    Datasets are stored in a central repository and indexed in a JSON file.
    Each dataset has metadata including size, checksum, format, tags, and
    registration timestamp.
    """

    def __init__(
        self,
        registry_dir: str = "/opt/datasets",
        index_filename: str = INDEX_FILENAME
    ):
        """
        Initialize the dataset registry.

        Args:
            registry_dir: Root directory for dataset storage
            index_filename: Name of the index file
        """
        self.registry_dir = Path(registry_dir)
        self.index_file = self.registry_dir / index_filename
        self._ensure_registry_exists()

    def _ensure_registry_exists(self) -> None:
        """Create registry directory and index file if they don't exist."""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_file.exists():
            self._save_index({})
        logger.info(f"Registry initialized at {self.registry_dir}")

    def _validate_path(self, path: str) -> None:
        """
        Validate that a path does not contain directory traversal attempts
        or absolute paths.

        Args:
            path: Path to validate

        Raises:
            ValueError: If path contains traversal sequences or is absolute
        """
        # Reject absolute paths
        if os.path.isabs(path):
            raise ValueError(f"Invalid path: absolute paths are not allowed '{path}'")

        normalized = os.path.normpath(path)
        if ".." in normalized.split(os.sep):
            raise ValueError(f"Invalid path: contains directory traversal sequence '{path}'")

    def _calculate_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        """
        Calculate checksum for a file.

        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use (sha256, sha1, md5)

        Returns:
            Hexadecimal checksum string
        """
        hash_func = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def _calculate_size_mb(self, file_path: Path) -> float:
        """
        Calculate file size in megabytes with one decimal place.

        Args:
            file_path: Path to the file

        Returns:
            Size in MB, rounded to one decimal place
        """
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 1)

    def _detect_format(self, file_path: Path) -> str:
        """
        Detect dataset format from file extension.

        Args:
            file_path: Path to the dataset file

        Returns:
            Format string (csv, parquet, json, excel, feather)
        """
        extension = file_path.suffix.lower()
        format_map = {
            ".csv": "csv",
            ".parquet": "parquet",
            ".json": "json",
            ".xlsx": "excel",
            ".xls": "excel",
            ".feather": "feather",
            ".fst": "feather",
        }
        return format_map.get(extension, "unknown")

    def _load_index(self) -> Dict[str, Any]:
        """Load the dataset index from JSON file."""
        try:
            with open(self.index_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse index file: {e}")
            return {}

    def _save_index(self, index: Dict[str, Any]) -> None:
        """Save the dataset index to JSON file."""
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)
        logger.debug(f"Index saved to {self.index_file}")

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO 8601 format."""
        return datetime.now(timezone.utc).isoformat()

    def register(
        self,
        source_path: str,
        dataset_name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> DatasetInfo:
        """
        Register a new dataset in the registry.

        Args:
            source_path: Path to the source file to register
            dataset_name: Unique name for the dataset
            description: Optional dataset description
            tags: Optional list of tags for categorization

        Returns:
            DatasetInfo object with registration details

        Raises:
            ValueError: If source file doesn't exist or path traversal detected
        """
        self._validate_path(source_path)
        source = Path(source_path)

        if not source.exists():
            raise ValueError(f"Source file does not exist: {source_path}")

        if not source.is_file():
            raise ValueError(f"Source path is not a file: {source_path}")

        logger.info(f"Registering dataset '{dataset_name}' from {source_path}")

        # Create dataset directory in registry
        dataset_dir = self.registry_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Copy file to registry
        dest_path = dataset_dir / source.name
        shutil.copy2(source, dest_path)
        logger.info(f"Copied dataset to {dest_path}")

        # Calculate metadata
        checksum = self._calculate_checksum(dest_path)
        size_mb = self._calculate_size_mb(dest_path)
        format_type = self._detect_format(dest_path)
        timestamp = self._get_timestamp()

        # Build source path in registry
        registry_source_path = str(dest_path)

        # Update index
        index = self._load_index()
        index[dataset_name] = {
            "name": dataset_name,
            "source_path": registry_source_path,
            "size_mb": size_mb,
            "checksum": checksum,
            "format": format_type,
            "description": description,
            "tags": tags or [],
            "registered_at": timestamp,
        }
        self._save_index(index)

        logger.info(f"Dataset '{dataset_name}' registered successfully")

        return DatasetInfo(
            name=dataset_name,
            source_path=registry_source_path,
            size_mb=size_mb,
            checksum=checksum,
            format=format_type,
            description=description,
            tags=tags or [],
            registered_at=timestamp,
        )

    def get(self, dataset_name: str) -> DatasetInfo:
        """
        Get dataset information by name.

        Args:
            dataset_name: Name of the dataset

        Returns:
            DatasetInfo object

        Raises:
            DatasetNotFoundError: If dataset not found
        """
        self._validate_path(dataset_name)
        index = self._load_index()

        if dataset_name not in index:
            raise DatasetNotFoundError(dataset_name)

        data = index[dataset_name]
        return DatasetInfo(**data)

    def list(
        self,
        name_contains: Optional[str] = None,
        tags: Optional[List[str]] = None,
        format_type: Optional[str] = None
    ) -> List[DatasetInfo]:
        """
        List datasets with optional filtering.

        Args:
            name_contains: Filter by name containing this string (case-insensitive)
            tags: Filter by tags (dataset must have all specified tags)
            format_type: Filter by format type

        Returns:
            List of matching DatasetInfo objects
        """
        index = self._load_index()
        results = []

        for data in index.values():
            dataset = DatasetInfo(**data)

            # Filter by name
            if name_contains and name_contains.lower() not in dataset.name.lower():
                continue

            # Filter by tags
            if tags:
                if not all(tag in dataset.tags for tag in tags):
                    continue

            # Filter by format
            if format_type and dataset.format != format_type:
                continue

            results.append(dataset)

        return sorted(results, key=lambda d: d.registered_at, reverse=True)

    def list_all(self) -> List[DatasetInfo]:
        """
        List all registered datasets.

        Returns:
            List of all DatasetInfo objects
        """
        return self.list()

    def search_by_name(self, query: str) -> List[DatasetInfo]:
        """
        Search datasets by name.

        Args:
            query: Search string

        Returns:
            List of matching DatasetInfo objects
        """
        return self.list(name_contains=query)

    def search_by_tags(self, tags: List[str]) -> List[DatasetInfo]:
        """
        Search datasets by tags.

        Args:
            tags: List of tags to match

        Returns:
            List of matching DatasetInfo objects
        """
        return self.list(tags=tags)

    def delete(self, dataset_name: str) -> bool:
        """
        Delete a dataset from the registry.

        Args:
            dataset_name: Name of the dataset to delete

        Returns:
            True if deleted, False if not found

        Raises:
            ValueError: If path traversal detected
        """
        self._validate_path(dataset_name)
        index = self._load_index()

        if dataset_name not in index:
            logger.warning(f"Dataset '{dataset_name}' not found for deletion")
            return False

        # Remove dataset directory
        dataset_dir = self.registry_dir / dataset_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            logger.info(f"Deleted dataset directory: {dataset_dir}")

        # Remove from index
        del index[dataset_name]
        self._save_index(index)

        logger.info(f"Dataset '{dataset_name}' deleted from registry")
        return True

    def exists(self, dataset_name: str) -> bool:
        """
        Check if a dataset exists in the registry.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if exists, False otherwise
        """
        self._validate_path(dataset_name)
        index = self._load_index()
        return dataset_name in index

    def verify_checksum(self, dataset_name: str) -> bool:
        """
        Verify that dataset file checksum matches the stored checksum.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if checksum matches, False otherwise

        Raises:
            DatasetNotFoundError: If dataset not found
        """
        index = self._load_index()
        if dataset_name not in index:
            raise DatasetNotFoundError(dataset_name)

        data = index[dataset_name]
        source_path = Path(data["source_path"])

        if not source_path.exists():
            logger.error(f"Dataset file missing: {source_path}")
            return False

        current_checksum = self._calculate_checksum(source_path)
        stored_checksum = data["checksum"]

        return current_checksum == stored_checksum

    def get_registry_path(self) -> str:
        """
        Get the registry directory path.

        Returns:
            Registry directory path as string
        """
        return str(self.registry_dir)

    def get_index_path(self) -> str:
        """
        Get the index file path.

        Returns:
            Index file path as string
        """
        return str(self.index_file)

    def prepare_for_workspace(
        self,
        dataset_name: str,
        workspace_dir: str,
        strategy: str = "copy"
    ) -> str:
        """
        Prepare a dataset for use in a workspace.

        Args:
            dataset_name: Name of the dataset
            workspace_dir: Workspace directory path
            strategy: Preparation strategy ('copy' or 'link')

        Returns:
            Path to the prepared dataset in workspace

        Raises:
            DatasetNotFoundError: If dataset not found
            ValueError: If invalid strategy specified
        """
        dataset = self.get(dataset_name)
        source = Path(dataset.source_path)
        dest = Path(workspace_dir) / source.name

        if strategy == "copy":
            if not dest.exists():
                shutil.copy2(source, dest)
                logger.info(f"Copied dataset to workspace: {dest}")
        elif strategy == "link":
            try:
                if not dest.exists():
                    os.link(source, dest)
                    logger.info(f"Linked dataset in workspace: {dest}")
            except OSError:
                # Fallback to copy if hard link fails (cross-filesystem)
                shutil.copy2(source, dest)
                logger.info(f"Copied dataset (link failed): {dest}")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return str(dest)

    def reload(self) -> None:
        """Reload the index from disk."""
        # Index is loaded on-demand, so this is a no-op
        logger.debug("Index reload requested (loaded on-demand)")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        index = self._load_index()
        total_size = sum(data["size_mb"] for data in index.values())

        # Count by format
        format_counts: Dict[str, int] = {}
        for data in index.values():
            format_counts[data["format"]] = format_counts.get(data["format"], 0) + 1

        # Count by tags
        tag_counts: Dict[str, int] = {}
        for data in index.values():
            for tag in data["tags"]:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_datasets": len(index),
            "total_size_mb": round(total_size, 1),
            "by_format": format_counts,
            "by_tags": tag_counts,
            "registry_path": str(self.registry_dir),
        }
