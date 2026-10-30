"""Check that subprocess result failures are logged in the parent run context."""

import pickle
import tempfile
import unittest
from pathlib import Path

from sparq.logging_config import logger, run_log_context
from sparq.tools.python_repl.executor import _execute_code_in_new_process


class TestExecutorLogging(unittest.TestCase):
    def test_result_write_failures_are_routed_to_the_owning_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            namespace_path = root / "namespace.pkl"
            namespace_path.write_bytes(pickle.dumps({}))
            other_run = root / "other-run"
            other_run.mkdir()
            records = []
            sink_id = logger.add(lambda message: records.append(message.record))
            try:
                with run_log_context(other_run, "other-run"):
                    for failure in ("missing", "directory"):
                        with self.subTest(failure=failure):
                            run_dir = root / failure
                            run_dir.mkdir()
                            result_path = run_dir / "result.json"
                            if failure == "directory":
                                result_path.mkdir()
                            else:
                                result_path = run_dir / "absent" / "result.json"

                            with run_log_context(run_dir, failure):
                                with logger.contextualize(node="executor"):
                                    result = _execute_code_in_new_process(
                                        "1 + 1",
                                        directory,
                                        timeout=10,
                                        ns_path=str(namespace_path),
                                        result_path=str(result_path),
                                    )

                            self.assertFalse(result.success)
                            self.assertIsNotNone(result.error)
                            assert result.error is not None
                            expected_error = (
                                "ResultMissingError"
                                if failure == "missing"
                                else "IsADirectoryError"
                            )
                            self.assertEqual(result.error.type, expected_error)
                            log_text = (run_dir / "log.txt").read_text()
                            self.assertIn(str(result_path), log_text)
                            self.assertIn("executor", log_text)
                            self.assertEqual(records[-1]["extra"]["run_id"], failure)
                            self.assertEqual(records[-1]["extra"]["node"], "executor")
                self.assertEqual((other_run / "log.txt").read_text(), "")
            finally:
                logger.remove(sink_id)


if __name__ == "__main__":
    unittest.main()
