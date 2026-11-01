"""Verify stable cache scope, default audit locations and cache preservation."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError
from rich.console import Console

from tests.utils.bedrock_utils.test_profiles import catalog_client
from utils.bedrock_utils import audit, paths


class TestPaths(unittest.TestCase):
    def test_model_directories_are_readable_without_merging_distinct_ids(self):
        self.assertEqual(
            paths.model_directory_name("us.openai.gpt-6.1-sol"), "us.openai.gpt-6.1-sol"
        )
        identifiers = [
            "example:v1",
            "example_v1",
            "Example_v1",
            "arn:aws:bedrock:region:account:application-inference-profile/app",
            "../outside",
            "x" * 300,
        ]
        names = [paths.model_directory_name(identifier) for identifier in identifiers]
        self.assertEqual(len(set(names)), len(identifiers))
        self.assertTrue(
            all(
                len(name) <= 174
                and "/" not in name
                and "\\" not in name
                and ":" not in name
                for name in names
            )
        )
        self.assertTrue(names[0].startswith("example_v1--"))
        self.assertEqual(names[0], paths.model_directory_name(identifiers[0]))

    def test_cache_uses_os_directory_and_keeps_contexts_distinct(self):
        with patch.object(paths, "user_cache_path", return_value=Path("/tmp/os-cache")):
            path = paths.profile_cache_path("sensd", "us-east-1")
        self.assertEqual(
            path,
            Path(
                "/tmp/os-cache/bedrock/sensd/us-east-1/inference_profiles.json"
            ).resolve(),
        )
        self.assertNotEqual(
            paths.context_parts("a/b", "us-east-1"),
            paths.context_parts("a_b", "us-east-1"),
        )
        self.assertEqual(paths.context_parts("..", "us-east-1")[0], "%2E%2E")

    def test_audit_outputs_live_beside_the_utility(self):
        with patch.object(
            paths, "__file__", "/tmp/project/utils/bedrock_utils/paths.py"
        ):
            self.assertEqual(
                paths.audit_context_dir("sensd", "us-east-1"),
                Path(
                    "/tmp/project/utils/bedrock_utils/outputs/sensd/us-east-1"
                ).resolve(),
            )

    def run_discovery(self, client, arguments):
        session = Mock(region_name="us-east-1")
        session.client.return_value = client
        with (
            patch("sparq.settings.ENVSettings"),
            patch.object(audit.boto3, "Session", return_value=session),
            patch.object(audit, "CONSOLE", Console(file=io.StringIO())),
        ):
            return audit.main(
                [
                    "--list-only",
                    "--profile",
                    "test",
                    "--region",
                    "us-east-1",
                    *arguments,
                ]
            )

    def test_incomplete_discovery_does_not_replace_a_complete_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "cache"
            self.assertEqual(
                self.run_discovery(
                    catalog_client(),
                    ["--cache-dir", str(cache_root), "--output", str(root / "first")],
                ),
                0,
            )
            cache = paths.profile_cache_path("test", "us-east-1", cache_root)
            original = cache.read_bytes()
            client = catalog_client()
            client.list_inference_profiles.side_effect = ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "Denied"}},
                "ListInferenceProfiles",
            )
            self.assertEqual(
                self.run_discovery(
                    client,
                    ["--cache-dir", str(cache_root), "--output", str(root / "second")],
                ),
                2,
            )
            self.assertEqual(cache.read_bytes(), original)
            report = json.loads((root / "second/report.json").read_text())
            self.assertIn("incomplete", report["cache_warning"])

    def test_default_audit_has_a_stable_latest_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / "utils/bedrock_utils/outputs/test/us-east-1"
            with patch.object(audit, "audit_context_dir", return_value=context):
                self.assertEqual(
                    self.run_discovery(
                        catalog_client(), ["--cache-dir", str(root / "cache")]
                    ),
                    0,
                )
            latest = json.loads((context / "latest_audit.json").read_text())
            self.assertTrue((context / latest["audit_dir"] / "report.json").is_file())
            self.assertTrue(latest["audit_dir"].startswith("audits/"))

    def test_shared_cache_writes_are_atomic_under_concurrent_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference_profiles.json"
            values = [audit.RunResult(reason=str(index)) for index in range(16)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda value: audit.atomic_json(path, value), values))
            self.assertIn(
                json.loads(path.read_text())["reason"], {str(i) for i in range(16)}
            )
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_script_entry_point_works_outside_the_checkout(self):
        script = Path(__file__).resolve().parents[3] / "scripts/aws/check_models.py"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--cache-dir", result.stdout)

    def test_module_and_legacy_script_entry_points_have_help(self):
        for command in (
            [sys.executable, "-m", "utils.bedrock_utils"],
            [sys.executable, "scripts/aws/check_models.py"],
        ):
            result = subprocess.run(
                [*command, "--help"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--cache-dir", result.stdout)
