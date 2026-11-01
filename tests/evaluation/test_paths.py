"""Verify shared evaluation destinations without invoking models."""

import tempfile
import unittest
from contextlib import chdir
from datetime import UTC, datetime, timedelta, timezone

from sparq.evaluation.paths import make_output_id, results_root
from sparq.utils.get_package_dir import get_project_root


class TestEvaluationPaths(unittest.TestCase):
    def test_checkout_destination_is_independent_of_working_directory(self):
        root = get_project_root()
        assert root is not None
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            self.assertEqual(results_root(), root / "evaluation/results")

    def test_directory_names_use_utc_and_do_not_collide_at_the_same_time(self):
        started = datetime(2026, 10, 31, 10, 30, tzinfo=timezone(timedelta(hours=-5)))
        first = make_output_id(started)
        second = make_output_id(started.astimezone(UTC))
        self.assertRegex(first, r"^2026-10-31T15-30-00Z_[0-9a-f]{8}$")
        self.assertNotEqual(first, second)
