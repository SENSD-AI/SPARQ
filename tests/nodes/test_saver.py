import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from pypdf import PdfReader

from sparq.architectures.v1.nodes.saver import saver_node
from sparq.architectures.v1.settings import V1LLMSettings
from sparq.schemas.state import State
from sparq.settings import LLMSetting


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
        markdown = (self.save_dir / 'report/report.md').read_text()
        self.assertIn("**User question**", markdown)
        self.assertIn("**Question asked:** Not recorded", markdown)
        self.assertTrue(markdown.strip().endswith(state.messages[-1].text))
        self.assertTrue((self.save_dir / 'report/report.pdf').is_file())

    def test_report_records_question_timestamps_and_distinct_models_in_both_formats(
        self,
    ):
        asked_at = datetime(2026, 10, 28, 14, 0, tzinfo=UTC)
        generated_at = datetime(2026, 10, 28, 14, 5, tzinfo=UTC)
        config = V1LLMSettings(
            **{
                role: LLMSetting(
                    model=f"{role}-model", provider="openai", recursion_limit=20
                )
                for role in (
                    "router",
                    "planner",
                    "executor",
                    "aggregator",
                    "vision_model",
                )
            }
        )
        state = State(
            query="Compare 2020 and 2022 county counts?",
            question_asked_at=asked_at,
            completed_plan_steps=[],
            route=True,
            messages=[
                AIMessage(
                    content="# County analysis\n\n## Results\n\nDifferent counts."
                )
            ],
        )
        with patch(
            "sparq.architectures.v1.nodes.saver.datetime", wraps=datetime
        ) as clock:
            clock.now.return_value = generated_at
            saver_node(state, self.save_dir, config)

        report_dir = self.save_dir / "report"
        markdown = (report_dir / "report.md").read_text()
        text = " ".join(
            " ".join(
                page.extract_text()
                for page in PdfReader(report_dir / "report.pdf").pages
            ).split()
        )
        for expected in (
            state.query,
            "2026-10-28T14:00:00+00:00",
            "2026-10-28T14:05:00+00:00",
            "Author: aggregator-model (openai) (aggregator)",
            "Planner model: planner-model (openai)",
            "Executor model: executor-model (openai)",
        ):
            self.assertIn(expected, text)
        self.assertLess(
            markdown.index("**User question**"), markdown.index("## Table of contents")
        )
        self.assertNotIn("router-model", text)

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
