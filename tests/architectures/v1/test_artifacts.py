"""Verify artifact categorization and preservation across repeated outputs."""

import tempfile
import unittest
from pathlib import Path

from sparq.architectures.v1.artifacts import Index, organize, organize_artifact


class TestArtifacts(unittest.TestCase):
    def test_repeated_outputs_are_preserved_and_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_paths = []
            for filename, content in (
                ("plot.png", "original"),
                ("plot_v1.png", "first duplicate"),
                ("plot_v2.png", "second duplicate"),
            ):
                (root / "plot.png").write_text(content)
                expected_path = Path("figures") / filename
                self.assertEqual(organize(root), [expected_path])
                self.assertEqual((root / expected_path).read_text(), content)
                expected_paths.append(expected_path)

            index = Index.model_validate_json((root / "index.json").read_text())
            self.assertEqual(
                index.figures,
                dict(zip(("f1", "f2", "f3"), expected_paths)),
            )
            self.assertIsNone(organize(root))
            for path, content in zip(
                expected_paths, ("original", "first duplicate", "second duplicate")
            ):
                self.assertEqual((root / path).read_text(), content)

    def test_categories_and_already_organized_files(self):
        for filename, category, artifact_id in (
            ("plot.PNG", "figures", "f1"),
            ("data.csv", "tables", "t1"),
            ("report.pdf", "docs", "d1"),
            ("data.bin", "other", "o1"),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / filename).write_text("content")
                index = Index()
                expected_path = Path(category) / filename
                self.assertEqual(
                    organize_artifact(index, Path(filename), root), expected_path
                )
                self.assertEqual(
                    organize_artifact(index, expected_path, root), expected_path
                )
                self.assertEqual(getattr(index, category), {artifact_id: expected_path})
                self.assertEqual((root / expected_path).read_text(), "content")
