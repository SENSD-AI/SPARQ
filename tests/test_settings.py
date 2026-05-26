import os
import unittest
from pathlib import Path

from sparq.settings import ENVSettings, BaseAgenticSettings


class TestENVSettings(unittest.TestCase):
    def setUp(self):
        self.settings = ENVSettings()

    def test_instantiates(self):
        self.assertIsInstance(self.settings, ENVSettings)

    def test_api_key_loaded(self):
        # At least one API key should be set
        keys = [
            self.settings.google_api_key,
            self.settings.gemini_api_key,
            self.settings.hf_token,
        ]
        self.assertTrue(any(keys), "Expected at least one API key to be loaded from .env")

    def test_no_unexpected_none_for_hf_token(self):
        self.assertIsNotNone(self.settings.hf_token, "HF_TOKEN should be set in .env")

    def test_langsmith_tracing_pushed_to_environ(self):
        self.assertEqual(os.environ.get("LANGSMITH_TRACING"), ("true" | "false"),
            "LANGSMITH_TRACING must be lowercase 'true' or lowercase 'false' in os.environ for LangSmith to enable tracing")


class TestBaseAgenticSettings(unittest.TestCase):
    def setUp(self):
        self.settings = BaseAgenticSettings()

    def test_instantiates(self):
        self.assertIsInstance(self.settings, BaseAgenticSettings)

    def test_test_query_loaded(self):
        self.assertIsInstance(self.settings.test_query, str)
        self.assertTrue(len(self.settings.test_query) > 0)

    def test_paths_loaded(self):
        self.assertIsNotNone(self.settings.paths.prompts_dir)
        self.assertIsNotNone(self.settings.paths.output_dir)

    def test_paths_are_path_objects(self):
        self.assertIsInstance(self.settings.paths.prompts_dir, Path)
        self.assertIsInstance(self.settings.paths.output_dir, Path)

    def test_paths_are_absolute(self):
        self.assertTrue(self.settings.paths.prompts_dir.is_absolute())
        self.assertTrue(self.settings.paths.output_dir.is_absolute())

    def test_run_dir(self):
        run_dir = self.settings.paths.run_dir
        self.assertIsInstance(run_dir, Path)
        self.assertTrue(run_dir.is_absolute())
        self.assertTrue(str(run_dir).startswith(str(self.settings.paths.output_dir)))


if __name__ == "__main__":
    unittest.main()
