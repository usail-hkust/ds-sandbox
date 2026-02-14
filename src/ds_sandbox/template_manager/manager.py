"""
Template management functionality for the sandbox manager.

This module provides the TemplateManager class for managing
workspace templates.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ds_sandbox.types import Template
from ds_sandbox.errors import SandboxError

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Template manager - handles template loading, saving, and lifecycle.

    This class manages workspace templates including:
    - Loading templates from disk
    - Saving templates to disk
    - Managing template aliases
    - Template CRUD operations
    """

    def __init__(self, templates_dir: Path):
        """
        Initialize the template manager.

        Args:
            templates_dir: Directory where templates are stored
        """
        self._templates_dir = templates_dir
        self._templates_dir.mkdir(parents=True, exist_ok=True)
        self._templates: Dict[str, Template] = {}
        self._template_aliases: Dict[str, str] = {}  # alias -> template_id

    def _load_templates(self) -> None:
        """Load templates from disk"""
        if not self._templates_dir.exists():
            return

        for template_file in self._templates_dir.glob("*.json"):
            try:
                with open(template_file, "r") as f:
                    template_data = json.load(f)
                    template = Template(**template_data)
                    self._templates[template.id] = template

                    # Register aliases
                    for alias in template.aliases:
                        self._template_aliases[alias] = template.id

                    logger.debug(f"Loaded template: {template.id}")
            except Exception as e:
                logger.warning(f"Failed to load template from {template_file}: {e}")

    def _save_template(self, template: Template) -> None:
        """Save template to disk"""
        template_file = self._templates_dir / f"{template.id}.json"
        with open(template_file, "w") as f:
            json.dump(template.model_dump(), f, indent=2)
        logger.debug(f"Saved template: {template.id}")

    async def build_template(
        self,
        template: Template,
        alias: Optional[str] = None,
        wait_timeout: int = 60,
        debug: bool = False,
    ) -> Template:
        """
        Build a template.

        This method stores the template configuration and optionally builds
        the actual Docker image if the backend supports it.

        Args:
            template: Template configuration
            alias: Primary alias for the template
            wait_timeout: Wait timeout in seconds
            debug: Enable debug mode

        Returns:
            Template object with assigned ID

        Example:
            >>> from ds_sandbox.template import TemplateBuilder
            >>> template = (TemplateBuilder()
            ...     .from_python_image("3.11")
            ...     .set_envs({"MY_VAR": "value"})
            ...     .build("my-template"))
            >>> manager = SandboxManager()
            >>> result = await manager.build_template(template)
        """
        # Set alias if provided
        if alias:
            if alias not in template.aliases:
                template.aliases.insert(0, alias)

        # Store template in memory
        self._templates[template.id] = template

        # Register aliases
        for template_alias in template.aliases:
            self._template_aliases[template_alias] = template.id

        # Save to disk
        self._save_template(template)

        logger.info(f"Template built: {template.id}")
        return template

    async def list_templates(self) -> List[Template]:
        """
        List all available templates.

        Returns:
            List of Template objects

        Example:
            >>> templates = await manager.list_templates()
            >>> for t in templates:
            ...     print(f"{t.id}: {t.name}")
        """
        return list(self._templates.values())

    async def get_template(self, template_id: str) -> Template:
        """
        Get template by ID or alias.

        Args:
            template_id: Template ID or alias

        Returns:
            Template object

        Raises:
            SandboxError: If template not found
        """
        # Try direct ID first
        if template_id in self._templates:
            return self._templates[template_id]

        # Try alias lookup
        if template_id in self._template_aliases:
            actual_id = self._template_aliases[template_id]
            return self._templates[actual_id]

        raise SandboxError(
            message=f"Template not found: {template_id}",
            error_code="TEMPLATE_NOT_FOUND",
        )

    async def delete_template(self, template_id: str) -> None:
        """
        Delete a template.

        Args:
            template_id: Template ID to delete

        Raises:
            SandboxError: If template not found
        """
        # Get template to remove aliases
        template = await self.get_template(template_id)

        # Remove from memory
        del self._templates[template_id]

        # Remove aliases
        for alias in template.aliases:
            self._template_aliases.pop(alias, None)

        # Remove from disk
        template_file = self._templates_dir / f"{template_id}.json"
        if template_file.exists():
            template_file.unlink()

        logger.info(f"Template deleted: {template_id}")
