"""Verify that a returned answer cannot hide a failed or bypassed SPARQ run."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sparq.evaluation import bedrock_probe as probe
from sparq.evaluation.bedrock_audit import ProbeJob, RunResult


def write_run(root: Path, trace: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "trace.json").write_text(json.dumps(trace))
    for name in [
        "result.json",
        "final_answer.json",
        "report/report.md",
        "report/report.pdf",
    ]:
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("synthetic artifact")


def successful_trace() -> dict:
    return {
        "route": True,
        "plan": {"steps": [{"id": 1}]},
        "results": [{"id": 1, "success": True}],
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


class TestWorker(unittest.IsolatedAsyncioTestCase):
    async def test_model_exception_becomes_failed_result(self):
        job = ProbeJob(
            model_id="bad",
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
            profile="test",
            region="us-east-1",
            query="test",
            artifact_dir=Path("/tmp/unused-probe"),
        )
        expected = RunResult(status="passed", reason="Completed")
        with patch.object(probe, "execute_job", new=AsyncMock(return_value=expected)):
            self.assertEqual(await probe.checked_job(job), expected)
