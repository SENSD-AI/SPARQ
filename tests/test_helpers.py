import json
import tempfile
import unittest
from pathlib import Path

from sparq.schemas.output_schemas import StepResult
from sparq.schemas.state import State
from sparq.utils.helpers import write_trace


class TestWriteTrace(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.save_dir = Path(self._tmp_dir.name)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_write_trace_writes_trace_json(self):
        # trace.json is written by write_trace() (called from Agentic_system.run()'s
        # astream loop on every node completion), not by saver_node itself, so a
        # crashed run still leaves the most recent trace on disk.
        state = State(
            query="What foods are most associated with Salmonella outbreaks?",
            answer="Poultry and eggs are the most commonly implicated food vehicles.",
            results=[StepResult(id=1, step="load data", success=True, execution_results="done")],
        )

        write_trace(self.save_dir, state)

        trace_path = self.save_dir / "trace.json"
        self.assertTrue(trace_path.exists())

        trace = json.loads(trace_path.read_text())
        self.assertEqual(trace["query"], state.query)
        self.assertEqual(trace["answer"], state.answer)
        self.assertEqual(len(trace["results"]), 1)
        self.assertEqual(trace["results"][0]["id"], 1)


if __name__ == "__main__":
    unittest.main()
