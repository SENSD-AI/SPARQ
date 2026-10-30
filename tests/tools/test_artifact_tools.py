"""Verify indexed discovery from executor and report working directories."""

import tempfile
import unittest
from pathlib import Path

from sparq.architectures.v1.artifacts import Index
from sparq.tools.artifact_tools import ArtifactTools


class TestArtifactTools(unittest.TestCase):
    def test_nested_index_returns_paths_relative_to_run_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = root / "executor"
            executor.mkdir()
            (executor / "index.json").write_text(
                Index(
                    figures={"f1": Path("figures/trend.png")},
                    tables={"t1": Path("tables/counts.csv")},
                ).model_dump_json()
            )
            tools = ArtifactTools(root, index_name="executor/index.json")

            self.assertEqual(
                tools.list_artifacts(),
                [
                    "executor/figures/trend.png",
                    "executor/tables/counts.csv",
                ],
            )
            self.assertEqual(
                tools.search_artifact("executor/figures/trend.png"),
                [
                    "executor/figures/trend.png",
                ],
            )
            worker_tools = ArtifactTools(executor)
            self.assertEqual(
                worker_tools.list_artifacts(),
                [
                    "figures/trend.png",
                    "tables/counts.csv",
                ],
            )

    def test_missing_index_has_no_indexed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            tools = ArtifactTools(Path(directory))
            self.assertEqual(tools.list_artifacts(), [])
            self.assertEqual(tools.search_artifact("trend.png"), [])


if __name__ == "__main__":
    unittest.main()
