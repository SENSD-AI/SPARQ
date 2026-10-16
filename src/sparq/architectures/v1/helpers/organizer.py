import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from sparq.schemas.state import State


class Index(BaseModel):
    figures: list[Path] = Field(default=[])
    tables: list[Path] = Field(default=[])
    docs: list[Path] = Field(default=[])
    uncategorized: list[Path] = Field(default=[])

INDEX_FNAME = 'index.json'

FIGURE_FTYPES = ['.png', '.jpg', '.jpeg', '.svg', '.gif', '.tif', '.tiff']
TABULAR_FTYPES = ['.csv', '.tsv', '.xls', '.xlsx', '.parquet']
DOC_FTYPES = ['.txt', '.md', '.pdf', '.doc', '.docx', '.html']

def organize(working_dir: Path) -> list[Path] | None:
    """Organizer artifacts produced so far

    - Put figures in figures/, tables in tables/, docs in docs/
    - Put rest of the artifacts in uncategorized/

    Check for any mismatches between the artifact index and existing artifacts.
    """
    # If no index file exists yet, create one
    index_path = working_dir / INDEX_FNAME
    if not index_path.exists():
        index_path.touch()
        index = Index()
    else:
        index = Index.model_validate_json(index_path.read_text())

    indexed_paths = set(
    index.figures
    + index.tables
    + index.docs
    + index.uncategorized
    )

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
        indexed_files = index.figures
    elif suffix in TABULAR_FTYPES:
        category = 'tables'
        indexed_files = index.tables
    elif suffix in DOC_FTYPES:
        category = 'docs'
        indexed_files = index.docs
    else:
        category = 'uncategorized'
        indexed_files = index.uncategorized

    source = working_dir / artifact_path
    relative_destination = Path(category) / artifact_path.name
    destination = working_dir / relative_destination

    if source != destination:
        if destination.exists():
            raise FileExistsError(f'Cannot overwrite existing artifact: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)

    if relative_destination not in indexed_files:
        indexed_files.append(relative_destination)

    return relative_destination
