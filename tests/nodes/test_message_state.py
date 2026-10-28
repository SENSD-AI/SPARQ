"""Verify message-based state updates and the existing single-run exports."""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader

from sparq.architectures.v1.nodes.router import router_node
from sparq.schemas.output_schemas import Router
from sparq.schemas.state import State
from sparq.settings import LLMSetting

with patch("tiktoken.get_encoding", return_value=Mock(encode=lambda text: list(text))):
    from sparq.architectures.v1.system import Agentic_system


class TestMessageState(unittest.TestCase):
    def test_analysis_route_does_not_append_a_provisional_answer(self):
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = Router(
            route=True, answer="Provisional text"
        )
        with patch("sparq.architectures.v1.nodes.router.get_llm", return_value=model):
            update = router_node(
                State(query="Analyze data"),
                LLMSetting(model="test-model", provider="openai"),
                "Route the request",
            )

        self.assertEqual(update, {"route": True, "messages": []})

    def test_message_updates_preserve_history_and_replace_matching_ids(self):
        graph = StateGraph(State)
        graph.add_node(
            "reply", lambda state: {"messages": [AIMessage(content="Revised", id="reply-1")]}
        )
        graph.add_edge(START, "reply")
        graph.add_edge("reply", END)

        result = graph.compile().invoke({
            "query": "A question",
            "messages": [
                HumanMessage(content="A question", id="question-1"),
                AIMessage(content="Original", id="reply-1"),
            ],
        })

        self.assertEqual([message.text for message in result["messages"]], ["A question", "Revised"])
        self.assertNotIn("answer", result)


class TestSingleRunMessageExport(unittest.IsolatedAsyncioTestCase):
    async def test_direct_answer_records_user_and_assistant_and_preserves_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = LLMSetting(model="test-model", provider="openai")
            llm_config = Mock(
                router=config, planner=config, executor=config,
                aggregator=config, vision_model=config,
            )
            llm_config.model_dump.return_value = {}
            system = Agentic_system.__new__(Agentic_system)
            system.settings = SimpleNamespace(
                paths=SimpleNamespace(run_dir=root), llm_config=llm_config,
            )
            system.system_prompts = {
                "router_prompt": "Route the request",
                "planner_prompt": "Plan",
                "executor_prompt": "Execute",
                "aggregator_prompt": "Report",
            }
            model = Mock()
            model.with_structured_output.return_value.invoke.return_value = Router(
                route=False, answer="A direct answer"
            )
            with (
                patch("sparq.architectures.v1.system.get_llm", return_value=Mock()),
                patch("sparq.architectures.v1.nodes.router.get_llm", return_value=model),
                patch("sparq.architectures.v1.system.cleanup_run"),
            ):
                output = await system.run("A question", run_id="test-run")

            self.assertEqual(output.response, "A direct answer")
            markdown = (root / "report/report.md").read_text()
            report_text = " ".join(
                page.extract_text()
                for page in PdfReader(root / "report/report.pdf").pages
            )
            self.assertIn("(router)", report_text)
            self.assertIn("**Planner model:** Not used", markdown)
            self.assertIn("**Executor model:** Not used", markdown)
            trace = json.loads((root / "trace.json").read_text())
            self.assertNotIn("answer", trace)
            asked_at = datetime.fromisoformat(trace["question_asked_at"])
            self.assertIsNotNone(asked_at.tzinfo)
            self.assertIn(asked_at.isoformat(timespec="seconds"), report_text)
            self.assertEqual(
                [message["type"] for message in trace["messages"]], ["human", "ai"]
            )
            self.assertEqual([message["content"] for message in trace["messages"]], ["A question", "A direct answer"])
            saved = json.loads((root / "final_answer.json").read_text())
            self.assertEqual(saved, {"query": "A question", "answer": "A direct answer"})
            result = json.loads((root / "result.json").read_text())
            self.assertEqual(result["response"], "A direct answer")
            self.assertEqual(result["run_id"], "test-run")


if __name__ == "__main__":
    unittest.main()
