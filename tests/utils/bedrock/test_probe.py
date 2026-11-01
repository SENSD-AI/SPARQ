"""Verify that a returned answer cannot hide a failed or bypassed SPARQ run."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sparq.utils.bedrock import probe
from sparq.utils.bedrock.audit import ProbeJob, RunResult
from sparq.utils.bedrock.coverage import REQUIRED_TOOLS, ToolEvidence


def coverage_calls() -> list[ToolEvidence]:
    calls = [ToolEvidence(name=name, output="ok") for name in sorted(REQUIRED_TOOLS)]
    for call in calls:
        if call.name == "load_dataset":
            call.inputs = {"file_path": "/data/input.csv"}
        elif call.name == "python_repl_tool":
            call.inputs = {"persist_namespace": True}
        elif call.name == "analyze_image":
            call.output = json.dumps(
                {"summary": "Visible plot", "answer": "Labels match"}
            )
    calls.append(
        ToolEvidence(
            name="load_dataset",
            inputs={"file_path": "/data/input.xlsx", "sheet_name": "Sheet1"},
            output="Loaded dataset",
        )
    )
    return calls


def write_run(root: Path, trace: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "trace.json").write_text(json.dumps(trace))
    for name in [
        "result.json",
        "final_answer.json",
        "report/report.md",
        "report/report.pdf",
        "report/ref.bib",
    ]:
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("synthetic artifact")
    (root / "report/report.md").write_text(
        "![Coverage chart](../executor/figures/plot.png)"
    )
    (root / "tool_calls.json").write_text(
        json.dumps([call.model_dump() for call in coverage_calls()])
    )
    executor = root / "executor"
    for category in ("figures", "tables", "docs"):
        (executor / category).mkdir(parents=True, exist_ok=True)
    (executor / "tables/feature_summary.csv").write_text(
        "dataset,rows,numeric_column,non_null_count,mean\n"
        "csv,10,value,9,2.5\nexcel,20,count,18,3\n"
    )
    (executor / "figures/plot.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (executor / "docs/methods.txt").write_text("source provenance")
    (executor / "docs/methods_copy.txt").write_text("source provenance")
    (executor / "docs/visual_check.txt").write_text("Labels match")
    (executor / "index.json").write_text(
        json.dumps(
            {
                "figures": {"f1": "figures/plot.png"},
                "tables": {"t1": "tables/feature_summary.csv"},
                "docs": {"d1": "docs/methods.txt"},
            }
        )
    )


def successful_trace() -> dict:
    return {
        "route": True,
        "plan": {
            "steps": [
                {"id": 1, "dependencies": []},
                {"id": 2, "dependencies": []},
                {"id": 3, "dependencies": [1, 2]},
            ]
        },
        "results": [{"id": i, "success": True} for i in (1, 2, 3)],
    }


class TestRunVerification(unittest.TestCase):
    def test_successful_analysis_with_outputs_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, successful_trace())
            probe.verify_run(root)

    def test_direct_answer_is_not_a_full_pipeline_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, {"route": False, "plan": None, "results": []})
            with self.assertRaisesRegex(ValueError, "did not exercise"):
                probe.verify_run(root)

    def test_failed_executor_step_preserves_its_reason(self):
        trace = successful_trace()
        trace["results"] = [
            {
                "id": 1,
                "success": False,
                "misc": "ValidationException: tool calling unsupported",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, trace)
            with self.assertRaisesRegex(ValueError, "tool calling unsupported"):
                probe.verify_run(root)

    def test_missing_report_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, successful_trace())
            (root / "report/report.pdf").unlink()
            with self.assertRaisesRegex(ValueError, "report.pdf"):
                probe.verify_run(root)

    def test_missing_feature_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, successful_trace())
            calls = [
                c.model_dump() for c in coverage_calls() if c.name != "analyze_image"
            ]
            (root / "tool_calls.json").write_text(json.dumps(calls))
            with self.assertRaisesRegex(ValueError, "analyze_image"):
                probe.verify_run(root)

    def test_single_step_cannot_claim_dependency_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = successful_trace()
            trace["plan"]["steps"] = [{"id": 1}]
            write_run(root, trace)
            with self.assertRaisesRegex(ValueError, "dependency merging"):
                probe.verify_run(root)

    def test_unbounded_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, successful_trace())
            table = root / "executor/tables/feature_summary.csv"
            table.write_text(table.read_text().replace("excel,20", "excel,201"))
            with self.assertRaisesRegex(ValueError, "bounded two-dataset"):
                probe.verify_run(root)

    def test_unembedded_plot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, successful_trace())
            (root / "report/report.md").write_text("Report without the saved plot")
            with self.assertRaisesRegex(ValueError, "embed an indexed PNG"):
                probe.verify_run(root)


class TestWorker(unittest.IsolatedAsyncioTestCase):
    async def test_model_exception_becomes_failed_result(self):
        job = ProbeJob(
            model_id="bad",
            mode="pipeline",
            profile="test",
            region="us-east-1",
            query="test",
            artifact_dir=Path("/tmp/unused-probe"),
        )
        with patch.object(
            probe,
            "execute_job",
            new=AsyncMock(side_effect=ValueError("Unsupported model")),
        ):
            result = await probe.checked_job(job)
        self.assertEqual(result.status, "failed")
        self.assertIn("Unsupported model", result.reason)

    async def test_worker_uses_candidate_for_text_roles_and_keeps_configured_vision(
        self,
    ):
        # Importing the production graph may otherwise fetch tokenizer data.
        with patch("tiktoken.get_encoding", return_value=Mock()):
            from sparq.architectures.v1 import system
        from sparq.architectures.v1.settings import V1Settings

        settings = V1Settings()  # pyright: ignore[reportCallIssue] -- TOML settings
        original_models = settings.llm_config.model_dump()
        captured = []

        class SyntheticSystem:
            def __init__(self, *, settings):
                self.settings = settings
                captured.append(settings)

            async def run(self, query):
                write_run(self.settings.paths.run_dir, successful_trace())
                return SimpleNamespace(response=query)

        with tempfile.TemporaryDirectory() as directory:
            job = ProbeJob(
                model_id="candidate",
                profile="test",
                region="us-east-1",
                query="synthetic query",
                artifact_dir=Path(directory),
            )
            with (
                patch(
                    "sparq.architectures.v1.settings.V1Settings", return_value=settings
                ),
                patch.object(system, "Agentic_system", SyntheticSystem),
                patch.object(
                    probe, "collect_tool_evidence", return_value=coverage_calls()
                ),
            ):
                result = await probe.execute_job(job)
        self.assertEqual(result.status, "passed")
        for role in ("router", "planner", "executor", "aggregator"):
            self.assertEqual(
                getattr(captured[0].llm_config, role).model_name, "candidate"
            )
            self.assertEqual(
                getattr(captured[0].llm_config, role).provider, "aws_bedrock"
            )
        self.assertEqual(
            captured[0].llm_config.vision_model, settings.llm_config.vision_model
        )
        self.assertEqual(settings.llm_config.model_dump(), original_models)

    async def test_successful_result_is_retained(self):
        job = ProbeJob(
            model_id="good",
            mode="pipeline",
            profile="test",
            region="us-east-1",
            query="test",
            artifact_dir=Path("/tmp/unused-probe"),
        )
        expected = RunResult(status="passed", reason="Completed")
        with patch.object(probe, "execute_job", new=AsyncMock(return_value=expected)):
            self.assertEqual(await probe.checked_job(job), expected)
