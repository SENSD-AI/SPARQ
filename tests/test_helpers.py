import json
import tempfile
import unittest
from pathlib import Path

from sparq.schemas.output_schemas import StepResult
from sparq.schemas.state import State
from sparq.utils.helpers import write_json


class TestWriteTrace(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.save_dir = Path(self._tmp_dir.name)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_write_trace_writes_trace_json(self):
        # trace.json is written by write_json() (called from Agentic_system.run()'s
        # astream loop on every node completion), not by saver_node itself, so a
        # crashed run still leaves the most recent trace on disk.
        state = State(
            query="What foods are most associated with Salmonella outbreaks?",
            answer="Poultry and eggs are the most commonly implicated food vehicles.",
            results=[StepResult(id=1, step="load data", success=True, execution_results="done")],
        )

        file_name = 'trace.json'
        write_json(self.save_dir, file_name, state)

        trace_path = self.save_dir / file_name
        self.assertTrue(trace_path.exists())

        trace = json.loads(trace_path.read_text())
        self.assertEqual(trace["query"], state.query)
        self.assertEqual(trace["answer"], state.answer)
        self.assertEqual(len(trace["results"]), 1)
        self.assertEqual(trace["results"][0]["id"], 1)

    def test_write_json_serializes_streamed_state_with_artifact_paths(self):
        state_values = {
            "query": "what datasets do we have",
            "results": [
                StepResult(
                    id=1,
                    step="list datasets",
                    success=True,
                    execution_results="done",
                    misc="",
                )
            ],
            "artifacts": [Path("docs/dataset_manifest_summary.txt")],
        }

        write_json(self.save_dir, "trace.json", state_values)

        trace = json.loads((self.save_dir / "trace.json").read_text())
        self.assertEqual(trace["artifacts"], ["docs/dataset_manifest_summary.txt"])
        self.assertEqual(trace["results"][0]["success"], True)

    def test_serialization_failure_preserves_previous_trace(self):
        from pydantic_core import PydanticSerializationError

        trace_path = self.save_dir / "trace.json"
        original = '{"query": "previous state"}'
        trace_path.write_text(original)

        with self.assertRaises(PydanticSerializationError):
            write_json(self.save_dir, "trace.json", {"unsupported": object()})

        self.assertEqual(trace_path.read_text(), original)


if __name__ == "__main__":
    unittest.main()
