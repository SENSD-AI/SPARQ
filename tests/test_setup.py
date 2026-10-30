"""Check first-run setup with isolated paths and downloads disabled."""

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sparq.architectures.v1.settings import V1_INNER_CONFIG

bootstrap = importlib.import_module("sparq.setup")


class TestSetup(unittest.TestCase):
    def test_setup_copies_canonical_summaries_without_downloading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            with (
                patch.object(bootstrap, "get_user_config_dir", return_value=root),
                patch.object(bootstrap, "USER_DATA_DIR", data_dir),
                patch.object(bootstrap, "USER_DOTFILE_PATH", root / ".env"),
                patch.object(
                    bootstrap,
                    "DATA_SUMMARIES_FULL_PATH",
                    data_dir / "data_summaries_full.json",
                ),
                patch.object(
                    bootstrap,
                    "DATA_SUMMARIES_SHORT_PATH",
                    data_dir / "data_summaries_short.json",
                ),
                patch.object(bootstrap, "SENTINEL", root / ".setup_complete"),
                patch.object(
                    bootstrap,
                    "ARCH_CONFIGS",
                    [(V1_INNER_CONFIG, root / "v1/config.toml")],
                ),
                patch("sparq.utils.download_data.main") as download,
            ):
                bootstrap.setup()

            self.assertEqual(
                (data_dir / "data_summaries_full.json").read_bytes(),
                (bootstrap.BUNDLED_DATA_DIR / "data_summaries_full.json").read_bytes(),
            )
            self.assertTrue((data_dir / "data_summaries_short.json").is_file())
            self.assertFalse((data_dir / "data_summaries.json").exists())
            self.assertTrue((root / ".setup_complete").is_file())
            self.assertTrue((root / "v1/config.toml").is_file())
            download.assert_called_once_with()
