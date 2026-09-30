import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

from eval.batch_eval import (
    Question, create_batch, execute_batch, execute_run, load_questions, main, parse_args, persist_manifest,
    update_batch_summary, write_json_atomic,
)
from sparq.architectures.v1.settings import V1Settings
from sparq.architectures.v1.system import Agentic_system
from sparq.schemas.output_schemas import BatchEvalOutput, BatchRunEntry, BatchStatus, RunStatus, SystemOutput


class TestBatchSummary(unittest.TestCase):
    def make_batch(self, statuses: list[str]) -> BatchEvalOutput:
        runs = [
            BatchRunEntry(
                question_id=index + 1,
                question_index=index,
                iteration=1,
                run_id=f"run-{index}",
                result_path=Path(f"runs/run-{index}/result.json"),
                status=cast(RunStatus, status),
            )
            for index, status in enumerate(statuses)
        ]
        return BatchEvalOutput(
            batch_id="batch-1",
            time_started=datetime.now(timezone.utc),
            dataset_path=Path("data/Q_dataset.json"),
            dataset_hash="sha256:test",
            question_filter={},
            requested_question_count=len(runs),
            selected_question_ids=[entry.question_id for entry in runs],
            iterations=1,
            max_concurrent_runs=3,
            code_version={},
            configuration={},
            planned_runs=len(runs),
            runs=runs,
        )

    def test_summary_is_completed_when_all_runs_complete(self):
        batch = self.make_batch(["completed", "completed"])

        update_batch_summary(batch)

        self.assertEqual(batch.status, "completed")
        self.assertEqual(batch.completed_runs, 2)

    def test_summary_reports_mixed_terminal_results(self):
        batch = self.make_batch(["completed", "failed"])

        update_batch_summary(batch)

        self.assertEqual(batch.status, "completed_with_errors")
        self.assertEqual(batch.completed_runs, 1)
        self.assertEqual(batch.failed_runs, 1)

    def test_inactive_batch_with_unfinished_runs_is_interrupted(self):
        batch = self.make_batch(["completed", "pending"])

        update_batch_summary(batch, active=False)

        self.assertEqual(batch.status, "interrupted")


class TestRunLifecycle(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.manifest_path = Path(self.temporary_directory.name) / "batch.json"
        self.entry = BatchRunEntry(
            question_id=1,
            question_index=0,
            iteration=1,
            run_id="run-1",
            result_path=Path("runs/iteration_001/question_001/run-1/result.json"),
        )
        self.batch = BatchEvalOutput(
            batch_id="batch-1",
            time_started=datetime.now(timezone.utc),
            dataset_path=Path("data/Q_dataset.json"),
            dataset_hash="sha256:test",
            question_filter={"weather_related": False},
            requested_question_count=1,
            selected_question_ids=[1],
            iterations=1,
            max_concurrent_runs=1,
            code_version={},
            configuration={},
            planned_runs=1,
            runs=[self.entry],
        )
        self.question: Question = {"id": 1, "text": "Test question?", "grade": 4, "weather_related": False}
        self.settings = V1Settings()  # pyright: ignore[reportCallIssue] -- fields come from TOML

    def tearDown(self):
        self.temporary_directory.cleanup()

    async def test_successful_run_is_persisted_as_completed(self):
        self.batch.evaluation_id = "baseline-v1"
        output = SystemOutput(
            run_id="run-1",
            query=self.question["text"],
            difficulty=4,
            models={},
            response="Test answer",
            time_started=datetime.now(timezone.utc),
            time_ended=datetime.now(timezone.utc),
            duration=0.1,
            ablation_config={},
            token_out=None,
            cost=None,
            evaluation_context=None,
            sparq_judge_score=None,
            sparq_judge_review=None,
        )
        agent = SimpleNamespace(run=AsyncMock(return_value=output))

        with patch("eval.batch_eval.Agentic_system", return_value=agent):
            await execute_run(
                asyncio.Semaphore(1), self.settings, self.question,
                self.entry, self.batch, self.manifest_path,
            )

        self.assertEqual(self.entry.status, "completed")
        evaluation_context = agent.run.await_args.kwargs["evaluation_context"]
        self.assertEqual(evaluation_context.batch_id, "batch-1")
        self.assertEqual(evaluation_context.evaluation_id, "baseline-v1")
        self.assertEqual(evaluation_context.question_id, 1)
        self.assertEqual(evaluation_context.iteration, 1)
        manifest = json.loads(self.manifest_path.read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["completed_runs"], 1)
        self.assertEqual(manifest["evaluation_id"], "baseline-v1")

    async def test_failed_run_records_the_exception(self):
        agent = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("model unavailable")))

        with patch("eval.batch_eval.Agentic_system", return_value=agent):
            await execute_run(
                asyncio.Semaphore(1), self.settings, self.question,
                self.entry, self.batch, self.manifest_path,
            )

        self.assertEqual(self.entry.status, "failed")
        self.assertEqual(self.entry.error_type, "RuntimeError")
        self.assertEqual(self.entry.error_message, "model unavailable")
        manifest = json.loads(self.manifest_path.read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failed_runs"], 1)

    async def test_pending_manifest_is_written_before_execution(self):
        persist_manifest(self.batch, self.manifest_path)

        manifest = json.loads(self.manifest_path.read_text())
        self.assertEqual(manifest["status"], "running")
        self.assertEqual(manifest["runs"][0]["status"], "pending")

    async def test_setup_failure_is_recorded(self):
        with patch("eval.batch_eval.Agentic_system", side_effect=ValueError("bad prompts")):
            await execute_run(
                asyncio.Semaphore(1), self.settings, self.question,
                self.entry, self.batch, self.manifest_path,
            )
        self.assertEqual(self.entry.status, "failed")
        self.assertEqual(self.entry.error_message, "bad prompts")

    def test_system_uses_supplied_settings_without_reloading(self):
        with patch.object(Agentic_system, "_load_prompts", return_value={}), patch(
            "sparq.architectures.v1.system.V1Settings",
        ) as loader:
            system = Agentic_system(settings=self.settings)
        loader.assert_not_called()
        self.assertIs(system.settings, self.settings)

    async def test_main_exit_status(self):
        for status, expected in (("completed", 0), ("completed_with_errors", 1), ("failed", 1)):
            with self.subTest(status=status):
                self.batch.status = cast(BatchStatus, status)
                with patch("eval.batch_eval.parse_args", return_value=SimpleNamespace(n_questions=1, iterations=1, evaluation_id=None)), patch(
                    "eval.batch_eval.load_questions", return_value=[self.question],
                ), patch("eval.batch_eval.ENVSettings"), patch(
                    "eval.batch_eval.V1Settings", return_value=self.settings,
                ), patch("eval.batch_eval.create_batch", return_value=(self.batch, self.manifest_path.parent)), patch(
                    "eval.batch_eval.execute_batch", new_callable=AsyncMock,
                ):
                    self.assertEqual(await main(), expected)

    def plan_runs(self):
        self.batch.runs = [
            self.entry.model_copy(update={
                "run_id": f"run-{iteration}-{index}",
                "iteration": iteration,
                "result_path": Path(f"runs/{iteration}-{index}/result.json"),
            })
            for iteration in (1, 2) for index in range(3)
        ]
        self.batch.planned_runs = 6
        self.batch.iterations = 2
        self.batch.max_concurrent_runs = 2

    async def test_concurrency_iteration_order_and_settings_isolation(self):
        self.plan_runs()
        active = 0
        peak = 0
        finished = []
        settings_seen = []

        async def run(*args, evaluation_context, **kwargs):
            nonlocal active, peak
            if evaluation_context.iteration == 2:
                self.assertEqual(finished.count(1), 3)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            finished.append(evaluation_context.iteration)

        def make_system(*, settings):
            settings_seen.append(settings)
            return SimpleNamespace(run=run)

        original_run_dir = self.settings.paths.run_dir
        with patch("eval.batch_eval.Agentic_system", side_effect=make_system):
            await execute_batch(self.batch, self.manifest_path.parent / "batch", [self.question], self.settings)
        self.assertEqual(peak, 2)
        self.assertEqual(self.batch.status, "completed")
        self.assertEqual(len({id(settings) for settings in settings_seen}), 6)
        self.assertEqual(self.settings.paths.run_dir, original_run_dir)

    async def test_cancellation_drains_running_and_waiting_runs(self):
        self.plan_runs()
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def run(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        batch_dir = self.manifest_path.parent / "batch"
        with patch("eval.batch_eval.Agentic_system", return_value=SimpleNamespace(run=run)):
            task = asyncio.create_task(execute_batch(self.batch, batch_dir, [self.question], self.settings))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(stopped.is_set())
        manifest = json.loads((batch_dir / "batch.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual(manifest["cancelled_runs"], 6)
        self.assertIsNotNone(manifest["time_ended"])

    async def test_run_failure_does_not_stop_batch(self):
        self.plan_runs()
        run = AsyncMock(side_effect=[RuntimeError("failed"), None, None, None, None, None])
        with patch("eval.batch_eval.Agentic_system", return_value=SimpleNamespace(run=run)):
            await execute_batch(self.batch, self.manifest_path.parent / "batch", [self.question], self.settings)
        self.assertEqual(self.batch.failed_runs, 1)
        self.assertEqual(self.batch.completed_runs, 5)
        self.assertEqual(self.batch.status, "completed_with_errors")

    async def test_manifest_failure_stops_and_drains_siblings(self):
        self.plan_runs()
        stopped = asyncio.Event()

        async def run(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        real_persist = persist_manifest
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("disk failure")
            real_persist(*args, **kwargs)

        batch_dir = self.manifest_path.parent / "batch"
        with patch("eval.batch_eval.Agentic_system", return_value=SimpleNamespace(run=run)), patch(
            "eval.batch_eval.persist_manifest", side_effect=fail_once,
        ):
            with self.assertRaisesRegex(OSError, "disk failure"):
                await execute_batch(self.batch, batch_dir, [self.question], self.settings)
        self.assertTrue(stopped.is_set())
        self.assertTrue(all(entry.status not in ("pending", "running") for entry in self.batch.runs))
        self.assertEqual(self.batch.status, "interrupted")


class TestInputAndPersistence(unittest.TestCase):
    def test_separate_batches_share_evaluation_id_and_keep_unique_artifacts(self):
        question: Question = {"id": 1, "text": "Question", "grade": 4, "weather_related": False}
        with patch("sys.argv", ["batch_eval", "-k", "1", "--evaluation-id", "baseline-v1"]):
            args = parse_args()
        with patch("eval.batch_eval.hash_file", return_value="sha256:test"), patch(
            "eval.batch_eval.get_code_version", return_value={},
        ):
            first, first_dir = create_batch([question], 1, args.iterations, {}, args.evaluation_id)
            second, second_dir = create_batch([question], 1, args.iterations, {}, args.evaluation_id)
        self.assertEqual(first.evaluation_id, "baseline-v1")
        self.assertEqual(first.evaluation_id, second.evaluation_id)
        self.assertNotEqual(first.batch_id, second.batch_id)
        self.assertNotEqual(first_dir, second_dir)
        self.assertNotEqual(first.runs[0].run_id, second.runs[0].run_id)
        legacy_manifest = first.model_dump(mode="json")
        del legacy_manifest["evaluation_id"]
        self.assertIsNone(BatchEvalOutput.model_validate(legacy_manifest).evaluation_id)

    def test_atomic_write_failure_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.json"
            write_json_atomic(path, {"old": True})
            with self.assertRaises(TypeError):
                write_json_atomic(path, {"invalid": object()})
            self.assertEqual(json.loads(path.read_text()), {"old": True})
            self.assertFalse(path.with_name(".batch.json.tmp").exists())

    def test_failed_replace_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.json"
            write_json_atomic(path, {"old": True})
            with patch.object(Path, "replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    write_json_atomic(path, {"new": True})
            self.assertEqual(json.loads(path.read_text()), {"old": True})
            self.assertFalse(path.with_name(".batch.json.tmp").exists())

    def test_question_validation_and_selection(self):
        question = {"id": 1, "text": "Question", "grade": 4, "weather_related": False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.json"
            write_json_atomic(path, {"questions": [question, {**question, "id": 2, "weather_related": True}]})
            self.assertEqual(load_questions(path, 1), [question])
            write_json_atomic(path, {"questions": [question, question]})
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_questions(path, 1)
            write_json_atomic(path, {"questions": [{**question, "grade": "4"}]})
            with self.assertRaisesRegex(ValueError, "Invalid question"):
                load_questions(path, 1)


if __name__ == "__main__":
    unittest.main()
