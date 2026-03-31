import unittest

from sparq.settings import ENVSettings, AgenticSystemSettings


class TestENVSettings(unittest.TestCase):
    def setUp(self):
        self.settings = ENVSettings()

    def test_instantiates(self):
        self.assertIsInstance(self.settings, ENVSettings)

    def test_api_key_loaded(self):
        # At least one API key should be set in the dev .env
        keys = [
            self.settings.google_api_key,
            self.settings.gemini_api_key,
            self.settings.hf_token,
        ]
        self.assertTrue(any(keys), "Expected at least one API key to be loaded from .env")

    def test_no_unexpected_none_for_hf_token(self):
        self.assertIsNotNone(self.settings.hf_token, "HF_TOKEN should be set in .env")


class TestAgenticSystemSettings(unittest.TestCase):
    def setUp(self):
        self.settings = AgenticSystemSettings()

    def test_instantiates(self):
        self.assertIsInstance(self.settings, AgenticSystemSettings)

    def test_test_query_loaded(self):
        self.assertIsInstance(self.settings.test_query, str)
        self.assertTrue(len(self.settings.test_query) > 0)

    def test_paths_loaded(self):
        self.assertIsNotNone(self.settings.paths.prompts_dir)
        self.assertIsNotNone(self.settings.paths.output_dir)

    def test_llm_config_loaded(self):
        # All four nodes should have an LLM config from the default TOML
        for node in ("router", "planner", "executor", "aggregator"):
            llm = getattr(self.settings.llm_config, node)
            self.assertIsNotNone(llm, f"Expected llm_config.{node} to be set")
            self.assertIsInstance(llm.model_name, str)
            self.assertIsInstance(llm.provider, str)


if __name__ == "__main__":
    unittest.main()
