"""Check download bookkeeping without network access or credential output."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sparq.utils import download_data


class TestDownloadData(unittest.TestCase):
    def test_download_uses_token_without_printing_it(self):
        token = "synthetic-sensitive-token"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            output = io.StringIO()
            manifest_path = root / "manifest.json"
            with (
                patch.object(
                    download_data,
                    "ENVSettings",
                    return_value=SimpleNamespace(hf_token=token),
                ),
                patch.object(download_data, "USER_DATA_DIR", root / "data"),
                patch.object(download_data, "DATA_MANIFEST_PATH", manifest_path),
                patch.object(download_data, "datasets", ["synthetic"]),
                patch.object(
                    download_data, "download_dataset_repo", return_value=str(cache)
                ) as download,
                redirect_stdout(output),
            ):
                download_data.main()

            self.assertNotIn(token, output.getvalue())
            download.assert_called_once_with(
                download_data.REPO_ID, "synthetic", token=token
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(
                Path(manifest["synthetic"]["cache_location"]).resolve(), cache.resolve()
            )
