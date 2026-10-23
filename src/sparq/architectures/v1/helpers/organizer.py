import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from sparq.schemas.state import State


class Index(BaseModel):
    figures: dict[str, Path] = Field(default_factory=dict)
    tables: dict[str, Path] = Field(default_factory=dict)
    docs: dict[str, Path] = Field(default_factory=dict)
    other: dict[str, Path] = Field(default_factory=dict)

INDEX_FNAME = 'index.json'

FIGURE_FTYPES = ['.png', '.jpg', '.jpeg', '.svg', '.gif', '.tif', '.tiff']
TABULAR_FTYPES = ['.csv', '.tsv', '.xls', '.xlsx', '.parquet']
DOC_FTYPES = ['.txt', '.md', '.pdf', '.doc', '.docx', '.html']

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


def organize_artifact(index: Index, artifact_path: Path, working_dir: Path)-> Path:
    """Move a relative artifact path into its category and update Index in memory.

    Paths stored in Index are relative to working_dir.
    """
    suffix = artifact_path.suffix.lower()
    if suffix in FIGURE_FTYPES:
        category = 'figures'
        subindex_letter = 'f'
        indexed_artifacts = index.figures
    elif suffix in TABULAR_FTYPES:
        category = 'tables'
        subindex_letter = 't'
        indexed_artifacts = index.tables
    elif suffix in DOC_FTYPES:
        category = 'docs'
        subindex_letter = 'd'
        indexed_artifacts = index.docs
    else:
        category = 'other'
        subindex_letter = 'o'
        indexed_artifacts = index.other

    source = working_dir / artifact_path
    relative_destination = Path(category) / artifact_path.name
    destination = working_dir / relative_destination

    if source != destination:
        if destination.exists():
            raise FileExistsError(f'Cannot overwrite existing artifact: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)

    if relative_destination not in indexed_artifacts.values():
        if indexed_artifacts:
            latest_subindex = list(indexed_artifacts.keys())[-1]
            latest_subindex_number = int(latest_subindex[1:])
        else:
            latest_subindex_number = 0
        new_subindex = subindex_letter + str(latest_subindex_number+1)
        indexed_artifacts[new_subindex] = relative_destination

    return relative_destination
