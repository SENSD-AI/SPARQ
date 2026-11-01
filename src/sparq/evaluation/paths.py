"""Shared locations and directory names for generated evaluation bundles."""

import uuid
from datetime import UTC, datetime
from pathlib import Path

from sparq.utils.get_package_dir import get_project_root


def results_root() -> Path:
    """Use the checkout's evaluation directory, or cwd for installed tools."""
    root = get_project_root() or Path.cwd()
    return root / "evaluation" / "results"


def make_output_id(started_at: datetime) -> str:
    """Create a sortable UTC directory name with a unique suffix."""
    timestamp = started_at.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"
