"""OS-native catalogue cache and project-local Bedrock audit evidence."""

import hashlib
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from platformdirs import user_cache_path


def context_parts(profile: str, region: str) -> tuple[str, str]:
    """Keep distinct AWS contexts in single path components, including unusual names."""
    parts = [
        "%2E" * len(value) if value in {".", ".."} else quote(value, safe="")
        for value in (profile, region)
    ]
    return parts[0], parts[1]


def audit_context_dir(profile: str, region: str) -> Path:
    return (
        Path(__file__).resolve().parent
        / "outputs"
        / Path(*context_parts(profile, region))
    )


def profile_cache_path(
    profile: str, region: str, cache_root: Path | None = None
) -> Path:
    root = (
        cache_root if cache_root is not None else user_cache_path("sparq") / "bedrock"
    )
    return (
        root.expanduser().resolve()
        / Path(*context_parts(profile, region))
        / "inference_profiles.json"
    )


def make_output_id(started_at: datetime) -> str:
    timestamp = started_at.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def model_directory_name(model_id: str) -> str:
    """Use a readable invocation ID, disambiguating sanitized or shortened names."""
    name = re.sub(r"[^a-z0-9._-]+", "_", model_id.lower()).strip("._-")
    name = name[:160].rstrip("._-") or "model"
    if name != model_id:
        digest = hashlib.sha256(model_id.encode()).hexdigest()[:12]
        return f"{name}--{digest}"
    return name
