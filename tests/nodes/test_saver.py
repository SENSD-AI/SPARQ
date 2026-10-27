import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

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
            messages=[
                HumanMessage(content="What foods are most associated with Salmonella outbreaks?"),
                AIMessage(content="Poultry and eggs are the most commonly implicated food vehicles."),
            ],
        )

        saver_node(state, save_dir=self.save_dir)

        final_answer_path = self.save_dir / "final_answer.json"
        self.assertTrue(final_answer_path.exists())

        final_answer = json.loads(final_answer_path.read_text())
        self.assertEqual(final_answer, {"query": state.query, "answer": state.messages[-1].text})
        self.assertEqual(
            (self.save_dir / 'report/report.md').read_text().strip(), state.messages[-1].text,
        )
        self.assertTrue((self.save_dir / 'report/report.pdf').is_file())

    def test_saves_query_and_answer_pair_when_answer_missing(self):
        state = State(query="What foods are most associated with Salmonella outbreaks?")

        saver_node(state, save_dir=self.save_dir)

        final_answer = json.loads((self.save_dir / "final_answer.json").read_text())
        self.assertEqual(final_answer, {"query": state.query, "answer": None})
        self.assertFalse((self.save_dir / 'report').exists())

    def test_user_message_is_not_saved_as_an_answer(self):
        state = State(query="A question", messages=[HumanMessage(content="A question")])

        saver_node(state, save_dir=self.save_dir)

        final_answer = json.loads((self.save_dir / "final_answer.json").read_text())
        self.assertIsNone(final_answer["answer"])

    def test_saves_text_blocks_without_reasoning_metadata(self):
        state = State(
            query="A question",
            messages=[AIMessage(content=[
                {"type": "reasoning", "reasoning": "Internal reasoning"},
                {"type": "text", "text": "The report"},
            ])],
        )

        saver_node(state, save_dir=self.save_dir)

        final_answer = json.loads((self.save_dir / "final_answer.json").read_text())
        self.assertEqual(final_answer["answer"], "The report")


if __name__ == "__main__":
    unittest.main()
