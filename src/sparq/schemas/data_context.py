import json
from pathlib import Path

from pydantic import BaseModel


class DataContext(BaseModel):
    manifest: dict
    summaries: dict

    def __str__(self) -> str:
        return (
            "=== Data Manifest ===\n"
            f"{json.dumps(self.manifest, indent=2)}\n\n"
            "=== Data Summaries ===\n"
            f"{json.dumps(self.summaries, indent=2)}"
        )


def load_data_context(manifest_path: Path, summaries_path: Path) -> DataContext:
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(summaries_path) as f:
        summaries = json.load(f)
    return DataContext(manifest=manifest, summaries=summaries)
