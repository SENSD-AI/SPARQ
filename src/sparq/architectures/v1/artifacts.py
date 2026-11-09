"""Index and organize executor artifacts by file category."""

from pathlib import Path

from pydantic import BaseModel, Field


class Index(BaseModel):
    figures: dict[str, Path] = Field(default_factory=dict)
    tables: dict[str, Path] = Field(default_factory=dict)
    docs: dict[str, Path] = Field(default_factory=dict)
    other: dict[str, Path] = Field(default_factory=dict)


INDEX_FNAME = "index.json"

FIGURE_FTYPES = [".png", ".jpg", ".jpeg", ".svg", ".gif", ".tif", ".tiff"]
TABULAR_FTYPES = [".csv", ".tsv", ".xls", ".xlsx", ".parquet"]
DOC_FTYPES = [".txt", ".md", ".pdf", ".doc", ".docx", ".html"]


def organize(working_dir: Path) -> list[Path] | None:
    """Organize unindexed artifacts by category and update the artifact index.

    Move files into figures/, tables/, docs/, or other/ based on extension.

    Args:
        working_dir: Directory containing the artifacts and index.json.

    Returns:
        Paths of newly indexed artifacts relative to working_dir, or None if
        no new artifacts are found.
    """
    # If no index file exists yet, create one and create an Index object. If it does exist, just write it into an Index object
    index_path = working_dir / INDEX_FNAME
    if not index_path.exists():
        index_path.touch()
        index = Index()
    else:
        index = Index.model_validate_json(index_path.read_text())

    # Collect all indexed artifacts
    indexed_paths: set[Path] = set()
    for category in (
        index.figures,
        index.tables,
        index.docs,
        index.other,
    ):
        indexed_paths.update(category.values())

    new_artifacts = []
    for artifact_path in list(working_dir.rglob("*")):
        if artifact_path.is_file() and artifact_path != index_path:
            relative_path = artifact_path.relative_to(working_dir)

            if relative_path not in indexed_paths:
                new_artifact = organize_artifact(index, relative_path, working_dir)
                new_artifacts.append(new_artifact)

    index_path.write_text(index.model_dump_json(indent=4))

    if new_artifacts:
        return new_artifacts

def versioned_path(filename: str | Path) -> Path:
    """Return the a versioned file name."""
    path = Path(filename)
    version = 1

    while True:
        candidate = path.with_stem(f"{path.stem}_v{version}")
        if not candidate.exists():
            return candidate
        version += 1


def organize_artifact(index: Index, artifact_path: Path, working_dir: Path) -> Path:
    """Move an artifact into its category, versioning filename collisions.

    Store and return the final path relative to working_dir.
    """
    suffix = artifact_path.suffix.lower()
    if suffix in FIGURE_FTYPES:
        category, id_prefix, indexed_artifacts = "figures", "f", index.figures
    elif suffix in TABULAR_FTYPES:
        category, id_prefix, indexed_artifacts = "tables", "t", index.tables
    elif suffix in DOC_FTYPES:
        category, id_prefix, indexed_artifacts = "docs", "d", index.docs
    else:
        category, id_prefix, indexed_artifacts = "other", "o", index.other

    source = working_dir / artifact_path
    destination = working_dir / category / artifact_path.name

    if source != destination:
        if destination.exists():
            destination = versioned_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)

    relative_destination = destination.relative_to(working_dir)
    if relative_destination in indexed_artifacts.values():
        return relative_destination

    next_id = 1
    if indexed_artifacts:
        last_id = list(indexed_artifacts)[-1]
        next_id = int(last_id[1:]) + 1
    indexed_artifacts[f"{id_prefix}{next_id}"] = relative_destination

    return relative_destination
