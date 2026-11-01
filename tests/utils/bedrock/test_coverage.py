"""Exercise local nested tracing and reject incomplete feature evidence."""

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.tracers.context import collect_runs

from sparq.tools.python_repl.schemas import OutputSchema
from sparq.utils.bedrock.coverage import (
    collect_tool_evidence,
    verify_tool_coverage,
)
from tests.utils.bedrock.test_probe import coverage_calls


class TestCoverage(unittest.TestCase):
    def check_calls(self, calls):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tool_calls.json").write_text(
                json.dumps([call.model_dump() for call in calls])
            )
            verify_tool_coverage(root)

    def test_tool_error_does_not_count_as_coverage(self):
        calls = coverage_calls()
        next(call for call in calls if call.name == "analyze_image").error = "Denied"
        with self.assertRaisesRegex(ValueError, "analyze_image"):
            self.check_calls(calls)

    def test_missing_excel_sheet_is_rejected(self):
        calls = coverage_calls()
        calls[-1].inputs.pop("sheet_name")
        with self.assertRaisesRegex(ValueError, "named Excel sheet"):
            self.check_calls(calls)

    def test_nonpersistent_python_is_rejected(self):
        calls = coverage_calls()
        next(call for call in calls if call.name == "python_repl_tool").inputs = {}
        with self.assertRaisesRegex(ValueError, "persistent Python"):
            self.check_calls(calls)

    def test_plain_text_vision_claim_is_rejected(self):
        calls = coverage_calls()
        next(
            call for call in calls if call.name == "analyze_image"
        ).output = "Looks fine"
        with self.assertRaisesRegex(ValueError, "structured visual"):
            self.check_calls(calls)


class TestLocalTracing(unittest.IsolatedAsyncioTestCase):
    async def test_failed_python_artifact_does_not_count_as_success(self):
        @tool(response_format="content_and_artifact")
        def python_probe() -> tuple[str, OutputSchema]:
            """Return an unsuccessful REPL result without running Python."""
            return "Execution failed", OutputSchema(
                output="", namespace={}, success=False, error=None
            )

        with collect_runs() as collector:
            await python_probe.ainvoke(
                {"name": "python_probe", "args": {}, "id": "test", "type": "tool_call"}
            )
        calls = collect_tool_evidence(collector.traced_runs)
        self.assertEqual(calls[0].error, "Execution failed")

    async def test_nested_tool_inputs_outputs_and_errors_are_collected(self):
        @tool
        def local_file_probe(path: str) -> str:
            """Inspect a fictional file without IO."""
            return f"Error: missing {path}"

        async def invoke_tool(path):
            return await local_file_probe.ainvoke({"path": path})

        with collect_runs() as collector:
            await RunnableLambda(invoke_tool).ainvoke("absent.csv")
        calls = collect_tool_evidence(collector.traced_runs)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].inputs, {"path": "absent.csv"})
        self.assertEqual(calls[0].error, "Error: missing absent.csv")
