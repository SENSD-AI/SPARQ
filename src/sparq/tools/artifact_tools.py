import difflib
from pathlib import Path

from ..architectures.v1.artifacts import Index

SEARCH_RESULT_LIMIT = 5
FUZZY_MATCHING_CUTOFF = 0.6


class ArtifactTools:
    def __init__(self, working_dir: Path, index_name: str = "index.json") -> None:
        self.working_dir = working_dir
        self.index_path = self.working_dir / index_name

    def search_artifact(self, artifact_name: str) -> list[str]:
        """Search for artifact in the artifact index
        and return the top 5 results by fuzzy matching.

        Returns:
        list[str]
        """
        # If an artifact name isn't provided, raise
        if artifact_name == "":
            return []

        all_artifacts = self.list_artifacts()
        matches = difflib.get_close_matches(
            artifact_name,
            all_artifacts,
            n=SEARCH_RESULT_LIMIT,
            cutoff=FUZZY_MATCHING_CUTOFF,
        )

        return matches

    def list_artifacts(self) -> list[str]:
        """Return all artifact paths in the index, relative to working_dir."""
        if not self.index_path.exists():
            return []
        index = Index.model_validate_json(self.index_path.read_text(encoding="utf-8"))
        # Index values are relative to the index's directory, which may be nested
        # below the filesystem tools' working directory (e.g. executor/index.json).
        index_directory = self.index_path.parent.relative_to(self.working_dir)

        return [
            str(index_directory / artifact_path)
            for field_name in Index.model_fields
            for artifact_path in getattr(index, field_name).values()
        ]
