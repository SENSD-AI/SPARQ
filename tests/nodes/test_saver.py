import json
import tempfile
import unittest
from pathlib import Path

from sparq.architectures.v1.nodes.saver import saver_node
from sparq.schemas.state import State


class TestSaverNode(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.save_dir = Path(self._tmp_dir.name)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_saves_query_and_answer_pair(self):
        state = State(
            query="What foods are most associated with Salmonella outbreaks?",
            answer="Poultry and eggs are the most commonly implicated food vehicles.",
        )

        saver_node(state, save_dir=self.save_dir)

        final_answer_path = self.save_dir / "final_answer.json"
        self.assertTrue(final_answer_path.exists())

        final_answer = json.loads(final_answer_path.read_text())
        self.assertEqual(final_answer, {"query": state.query, "answer": state.answer})

    def test_saves_query_and_answer_pair_when_answer_missing(self):
        state = State(query="What foods are most associated with Salmonella outbreaks?")

        saver_node(state, save_dir=self.save_dir)

        final_answer = json.loads((self.save_dir / "final_answer.json").read_text())
        self.assertEqual(final_answer, {"query": state.query, "answer": None})


if __name__ == "__main__":
    unittest.main()
