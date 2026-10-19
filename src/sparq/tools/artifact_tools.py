import difflib
import json
from pathlib import Path

from langchain.tools import tool

from ..architectures.v1.helpers.organizer import Index

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

        # Load artifact index
        index = json.loads(self.index_path.read_text())  # dict
        index = Index.model_validate(index)             # Index instance

        all_artifacts = []
        for field_name, field_info in Index.model_fields.items():
            value = getattr(index, field_name)

            for artifact_path in value:
                all_artifacts.append(str(artifact_path))

        matches = difflib.get_close_matches(artifact_name, all_artifacts, n=SEARCH_RESULT_LIMIT, cutoff=FUZZY_MATCHING_CUTOFF) 

        return matches

    def list_artifacts(self) -> list[str]:
        """Return all artifact paths in the index, relative to working_dir."""
        index = Index.model_validate_json(self.index_path.read_text())

        return [
            str(artifact_path)
            for field_name in Index.model_fields
            for artifact_path in getattr(index, field_name)
        ]
