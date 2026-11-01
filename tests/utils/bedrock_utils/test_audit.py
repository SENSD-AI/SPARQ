"""Exercise catalog coverage, per-model errors and checkpointing without AWS calls."""

import csv
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from threading import Barrier, Event, Lock
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError
from rich.console import Console

from utils.bedrock_utils import audit


def foundation(identifier: str, output: str = "TEXT") -> dict:
    return {
        "modelId": identifier,
        "modelName": identifier,
        "inputModalities": ["TEXT"],
        "outputModalities": [output],
        "inferenceTypesSupported": ["ON_DEMAND"],
    }


def target(identifier: str) -> audit.ModelResult:
    return audit.ModelResult(
        model_id=identifier,
        name=identifier,
        kind="foundation",
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        inference_types=["ON_DEMAND"],
    )


class TestDiscovery(unittest.TestCase):
    def test_foundations_and_all_profile_pages_are_preserved(self):
        client = Mock()
        client.list_foundation_models.return_value = {
            "modelSummaries": [foundation("text"), foundation("embed", "EMBEDDING")]
        }
        client.list_inference_profiles.side_effect = [
            {
                "inferenceProfileSummaries": [
                    {
                        "inferenceProfileId": "us.text",
                        "models": [
                            {
                                "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/text"
                            }
                        ],
                    }
                ],
                "nextToken": "page-2",
            },
            {
                "inferenceProfileSummaries": [
                    {
                        "inferenceProfileId": "app",
                        "inferenceProfileArn": "arn:application",
                        "type": "APPLICATION",
                        "models": [],
                    }
                ]
            },
        ]
        models, errors = audit.discover_models(client)
        self.assertEqual(
            {model.model_id for model in models},
            {"text", "embed", "us.text", "arn:application"},
        )
        self.assertEqual(errors, [])
        client.list_foundation_models.assert_called_once_with()
        self.assertEqual(
            client.list_inference_profiles.call_args_list[1].kwargs,
            {"nextToken": "page-2"},
        )
        profile = next(model for model in models if model.model_id == "us.text")
        self.assertEqual(profile.foundation_model_ids, ["text"])
        self.assertEqual(profile.input_modalities, ["TEXT"])
        self.assertEqual(profile.output_modalities, ["TEXT"])
        self.assertEqual(profile.inference_types, ["ON_DEMAND"])

    def test_discovery_error_retains_partial_catalog(self):
        client = Mock()
        client.list_foundation_models.return_value = {
            "modelSummaries": [foundation("text")]
        }
        client.list_inference_profiles.side_effect = ClientError(
            {
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": "No profile-list permission",
                }
            },
            "ListInferenceProfiles",
        )
        models, errors = audit.discover_models(client)
        self.assertEqual(len(models), 1)
        self.assertIn("AccessDeniedException", errors[0])

    def test_known_non_text_models_and_unknown_metadata_are_excluded(self):
        model = target("embed")
        model.output_modalities = ["EMBEDDING"]
        reason = audit.exclusion_reason(model)
        assert reason is not None
        self.assertIn("TEXT output", reason)
        unknown = target("unknown-profile")
        unknown.input_modalities = None
        self.assertIn("Cannot verify", audit.exclusion_reason(unknown) or "")

    def test_wrapped_aws_error_keeps_original_code_and_message(self):
        original = ClientError(
            {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "Use an inference profile",
                }
            },
            "Converse",
        )
        wrapper = ValueError("Could not initialize model")
        wrapper.__cause__ = original
        result = audit.failure_result(wrapper)
        self.assertEqual(result.error_code, "ValidationException")
        self.assertIn("Use an inference profile", result.reason)
        self.assertEqual(len(result.error_chain), 2)


class TestSweep(unittest.TestCase):
    def test_saved_artifact_paths_survive_moving_the_sweep(self):
        report = audit.AuditReport(
            profile="test", region="us-east-1", models=[target("example-model")]
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sweep"
            run_dir = output / "runs" / "example-model"
            run_dir.mkdir(parents=True)
            (run_dir / "worker.log").write_text("synthetic evidence")
            with patch.object(
                audit,
                "run_probe",
                return_value=audit.RunResult(
                    status="passed",
                    artifact_dir=str(run_dir),
                ),
            ):
                audit.run_models(report, output, timeout=10)

            moved = output.rename(Path(directory) / "archived-sweep")
            saved = audit.AuditReport.model_validate_json(
                (moved / "report.json").read_text()
            )
            relative = saved.models[0].run.artifact_dir
            self.assertEqual(relative, "runs/example-model")
            self.assertEqual(
                (moved / str(relative) / "worker.log").read_text(), "synthetic evidence"
            )
            with (moved / "results.csv").open(newline="") as stream:
                self.assertEqual(next(csv.DictReader(stream))["artifact_dir"], relative)

    def test_failed_model_does_not_stop_next_model_and_reports_are_written(self):
        report = audit.AuditReport(
            profile="test", region="us-east-1", models=[target("bad"), target("good")]
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(
                audit,
                "run_probe",
                side_effect=[
                    audit.RunResult(status="failed", reason="Access denied"),
                    audit.RunResult(status="passed", reason="Completed"),
                ],
            ):
                self.assertTrue(audit.run_models(report, output, timeout=10))
            saved = audit.AuditReport.model_validate_json(
                (output / "report.json").read_text()
            )
            self.assertEqual(
                [model.run.status for model in saved.models], ["failed", "passed"]
            )
            summary = (output / "summary.md").read_text()
            self.assertIn("`good`", summary)
            self.assertIn("Access denied", summary)
            with (output / "results.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_interrupt_retains_completed_results_and_untested_target(self):
        report = audit.AuditReport(
            profile="test",
            region="us-east-1",
            models=[target("one"), target("two"), target("three")],
        )
        original_completed = audit.as_completed
        second_running = Event()

        def completed_then_interrupt(futures):
            yield next(original_completed(futures))
            self.assertTrue(second_running.wait(3))
            raise KeyboardInterrupt()

        def probe(job, timeout, cancel):
            if cancel.is_set():
                return audit.RunResult(reason="Interrupted before start")
            if job.model_id == "one":
                return audit.RunResult(status="passed", reason="Completed")
            second_running.set()
            self.assertTrue(
                cancel.wait(3), "Running worker did not receive cancellation"
            )
            return audit.RunResult(status="failed", error_code="RunInterrupted")

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(audit, "run_probe", side_effect=probe),
                patch.object(
                    audit, "as_completed", side_effect=completed_then_interrupt
                ),
            ):
                self.assertFalse(
                    audit.run_models(report, Path(directory), 10, workers=1)
                )
            saved = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(saved["models"][0]["run"]["status"], "passed")
            self.assertEqual(saved["models"][1]["run"]["error_code"], "RunInterrupted")
            self.assertFalse(
                Path(saved["models"][1]["run"]["artifact_dir"]).is_absolute()
            )
            self.assertEqual(saved["models"][2]["run"]["status"], "not_tested")

    def test_hard_timeout_records_failure_and_stops_worker(self):
        process = Mock()
        process.wait.side_effect = subprocess.TimeoutExpired("probe", 1)
        with tempfile.TemporaryDirectory() as directory:
            job = audit.ProbeJob(
                model_id="model",
                profile="test",
                region="us-east-1",
                query="test",
                artifact_dir=Path(directory),
            )
            with (
                patch.object(audit.subprocess, "Popen", return_value=process),
                patch.object(audit, "stop_worker") as stop,
            ):
                result = audit.run_probe(job, 1)
            self.assertEqual(result.error_code, "RunTimeout")
            self.assertEqual(result.status, "timed_out")
            stop.assert_called_once_with(process)
            self.assertEqual(result.artifact_dir, directory)

    def test_worker_exit_without_result_is_not_marked_usable(self):
        process = Mock(returncode=1)
        with tempfile.TemporaryDirectory() as directory:
            job = audit.ProbeJob(
                model_id="model",
                profile="test",
                region="us-east-1",
                query="test",
                artifact_dir=Path(directory),
            )
            with patch.object(
                audit.subprocess, "Popen", return_value=process
            ) as launch:
                result = audit.run_probe(job, 10)
            arguments, keywords = launch.call_args
            self.assertEqual(arguments[0][1:3], ["-m", "utils.bedrock_utils.probe"])
            self.assertEqual(
                keywords["env"]["PYTHONPATH"].split(os.pathsep)[0],
                str(Path(audit.__file__).resolve().parents[2]),
            )
            self.assertEqual(result.error_code, "WorkerFailure")
            self.assertEqual(result.status, "failed")

    def test_hard_timeout_keeps_finished_capability_checks(self):
        process = Mock()
        process.wait.side_effect = subprocess.TimeoutExpired("probe", 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = audit.ProbeJob(
                model_id="test",
                profile="test",
                region="us-east-1",
                query="",
                artifact_dir=root,
            )
            partial = audit.RunResult(
                active_check="structured_plan",
                checks={
                    "structured_router": audit.CheckResult(status="passed"),
                    "structured_plan": audit.CheckResult(),
                    "tool_round_trip": audit.CheckResult(),
                },
            )
            audit.atomic_json(root / "check.json", partial)
            with (
                patch.object(audit.subprocess, "Popen", return_value=process),
                patch.object(audit, "stop_worker"),
            ):
                result = audit.run_probe(job, 1)
            self.assertEqual(result.status, "timed_out")
            self.assertEqual(result.checks["structured_router"].status, "passed")
            self.assertEqual(result.checks["structured_plan"].status, "timed_out")
            self.assertEqual(result.checks["tool_round_trip"].status, "skipped")

    def test_list_only_uses_configured_region_without_running_models(self):
        session = Mock(region_name="us-east-1")
        session.client.return_value.list_foundation_models.return_value = {
            "modelSummaries": [foundation("text"), foundation("embedding", "EMBEDDING")]
        }
        session.client.return_value.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": []
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("sparq.settings.ENVSettings"),
                patch.dict(
                    audit.os.environ, {"AWS_REGION": "us-east-1", "AWS_PROFILE": "test"}
                ),
                patch.object(audit.boto3, "Session", return_value=session),
                patch.object(audit, "run_models") as run,
            ):
                code = audit.main(
                    [
                        "--list-only",
                        "--cache-dir",
                        str(Path(directory) / "cache"),
                        "--output",
                        directory,
                    ]
                )
            run.assert_not_called()
            saved = audit.AuditReport.model_validate_json(
                (Path(directory) / "report.json").read_text()
            )
            self.assertEqual(code, 0)
            self.assertEqual(saved.region, "us-east-1")
            self.assertEqual(
                [model.run.status for model in saved.models], ["skipped", "not_tested"]
            )


class TestDefaultCommand(unittest.TestCase):
    def test_default_command_runs_every_text_target_and_saves_outcomes(self):
        session = Mock(region_name="us-east-1")
        client = session.client.return_value
        client.list_foundation_models.return_value = {
            "modelSummaries": [
                foundation("bad"),
                foundation("good"),
                foundation("embed", "EMBEDDING"),
            ]
        }
        client.list_inference_profiles.return_value = {"inferenceProfileSummaries": []}
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("sparq.settings.ENVSettings"),
                patch.dict(
                    audit.os.environ, {"AWS_PROFILE": "test", "AWS_REGION": "us-east-1"}
                ),
                patch.object(audit.boto3, "Session", return_value=session),
                patch.object(
                    audit,
                    "run_probe",
                    side_effect=lambda job, timeout, cancel: audit.RunResult(
                        status="failed" if job.model_id == "bad" else "passed",
                        reason="Unsupported tools"
                        if job.model_id == "bad"
                        else "Completed",
                    ),
                ) as probe,
            ):
                self.assertEqual(
                    audit.main(
                        [
                            "--cache-dir",
                            str(Path(directory) / "cache"),
                            "--output",
                            directory,
                        ]
                    ),
                    0,
                )
            self.assertEqual(
                [call.args[0].model_id for call in probe.call_args_list],
                ["bad", "good"],
            )
            report = audit.AuditReport.model_validate_json(
                (Path(directory) / "report.json").read_text()
            )
            self.assertEqual(
                [row.run.status for row in report.models],
                ["failed", "skipped", "passed"],
            )
            self.assertIn(
                "Unsupported tools", (Path(directory) / "summary.md").read_text()
            )


class TestEligibility(unittest.TestCase):
    def test_text_and_on_demand_are_required_but_images_are_optional(self):
        cases = [
            ({}, None),
            (
                {
                    "input_modalities": ["TEXT", "IMAGE"],
                    "output_modalities": ["TEXT", "IMAGE"],
                },
                None,
            ),
            ({"inference_types": ["PROVISIONED", "ON_DEMAND"]}, None),
            ({"input_modalities": ["IMAGE"]}, "TEXT input"),
            ({"output_modalities": ["IMAGE"]}, "TEXT output"),
            ({"inference_types": ["PROVISIONED"]}, "ON_DEMAND throughput"),
            ({"inference_types": None}, "Cannot verify"),
            ({"input_modalities": None}, "Cannot verify"),
            ({"output_modalities": None}, "Cannot verify"),
        ]
        for fields, expected in cases:
            with self.subTest(fields=fields):
                model = target("example").model_copy(update=fields)
                reason = audit.exclusion_reason(model)
                if expected is None:
                    self.assertIsNone(reason)
                else:
                    self.assertIn(expected, reason or "")

    def test_profiles_require_eligible_metadata_for_every_base(self):
        provisioned = foundation("provisioned")
        provisioned["inferenceTypesSupported"] = ["PROVISIONED"]
        multimodal = foundation("multimodal")
        multimodal["inputModalities"] = ["TEXT", "IMAGE"]
        client = Mock()
        client.list_foundation_models.return_value = {
            "modelSummaries": [foundation("text"), provisioned, multimodal]
        }
        client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "good",
                    "models": [{"modelArn": "arn:foundation-model/text"}],
                },
                {
                    "inferenceProfileId": "images",
                    "models": [{"modelArn": "arn:foundation-model/multimodal"}],
                },
                {
                    "inferenceProfileId": "mixed",
                    "models": [
                        {"modelArn": "arn:foundation-model/text"},
                        {"modelArn": "arn:foundation-model/provisioned"},
                    ],
                },
                {
                    "inferenceProfileId": "unknown",
                    "models": [{"modelArn": "arn:foundation-model/absent"}],
                },
            ],
        }
        models, errors = audit.discover_models(client)
        self.assertEqual(errors, [])
        profiles = {
            model.model_id: model
            for model in models
            if model.kind == "inference_profile"
        }
        self.assertIsNone(audit.exclusion_reason(profiles["good"]))
        self.assertIsNone(audit.exclusion_reason(profiles["images"]))
        self.assertEqual(profiles["images"].input_modalities, ["IMAGE", "TEXT"])
        self.assertIn("ON_DEMAND", audit.exclusion_reason(profiles["mixed"]) or "")
        self.assertIn(
            "Cannot verify", audit.exclusion_reason(profiles["unknown"]) or ""
        )

    def test_sweep_never_invokes_provisioned_only_or_unknown_models(self):
        report = audit.AuditReport(
            profile="test",
            region="us-east-1",
            models=[
                target("text"),
                target("multimodal").model_copy(
                    update={"input_modalities": ["TEXT", "IMAGE"]}
                ),
                target("provisioned").model_copy(
                    update={"inference_types": ["PROVISIONED"]}
                ),
                target("unknown").model_copy(update={"inference_types": None}),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                audit,
                "run_probe",
                return_value=audit.RunResult(status="passed", reason="Completed"),
            ) as probe:
                audit.run_models(report, Path(directory), 10)
            self.assertEqual(
                [call.args[0].model_id for call in probe.call_args_list],
                ["text", "multimodal"],
            )
            saved = audit.AuditReport.model_validate_json(
                (Path(directory) / "report.json").read_text()
            )
            self.assertEqual(
                [model.run.status for model in saved.models],
                ["passed", "passed", "skipped", "skipped"],
            )
            self.assertIn("ON_DEMAND", saved.models[2].run.reason)


class TestParallelSweep(unittest.TestCase):
    def test_runs_overlap_without_exceeding_worker_limit(self):
        report = audit.AuditReport(
            profile="test",
            region="us-east-1",
            models=[target("one"), target("two"), target("three")],
        )
        lock = Lock()
        together = Barrier(2)
        active = 0
        peak = 0
        checkpoint_threads = []
        original_save = audit.save_report

        def probe(job, timeout, cancel):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if job.model_id in {"one", "two"}:
                    together.wait(timeout=3)
                return audit.RunResult(status="passed", reason="Completed")
            finally:
                with lock:
                    active -= 1

        def save(report, output):
            import threading

            checkpoint_threads.append(threading.get_ident())
            original_save(report, output)

        import threading

        coordinator = threading.get_ident()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(audit, "run_probe", side_effect=probe),
            patch.object(audit, "save_report", side_effect=save),
        ):
            self.assertTrue(audit.run_models(report, Path(directory), 10, workers=2))
            saved = audit.AuditReport.model_validate_json(
                (Path(directory) / "report.json").read_text()
            )
        self.assertEqual(peak, 2)
        self.assertEqual([model.run.status for model in saved.models], ["passed"] * 3)
        self.assertEqual(set(checkpoint_threads), {coordinator})

    def test_fast_run_is_checkpointed_without_waiting_for_slow_run(self):
        report = audit.AuditReport(
            profile="test", region="us-east-1", models=[target("slow"), target("fast")]
        )
        fast_saved = Event()
        original_save = audit.save_report

        def probe(job, timeout, cancel):
            if job.model_id == "slow":
                self.assertTrue(
                    fast_saved.wait(3), "Faster result was blocked by input order"
                )
            return audit.RunResult(status="passed", reason="Completed")

        def save(report, output):
            original_save(report, output)
            if report.models[1].run.status == "passed":
                fast_saved.set()

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(audit, "run_probe", side_effect=probe),
            patch.object(audit, "save_report", side_effect=save),
        ):
            self.assertTrue(audit.run_models(report, Path(directory), 10, workers=2))
        self.assertTrue(fast_saved.is_set())

    def test_interrupt_cancels_every_active_worker(self):
        report = audit.AuditReport(
            profile="test",
            region="us-east-1",
            models=[target(str(i)) for i in range(5)],
        )
        running = Barrier(3)
        started = []
        lock = Lock()

        def probe(job, timeout, cancel):
            if cancel.is_set():
                return audit.RunResult(reason="Interrupted before start")
            with lock:
                started.append(job.model_id)
            running.wait(timeout=3)
            self.assertTrue(cancel.wait(3), "Running worker was left behind")
            return audit.RunResult(status="failed", error_code="RunInterrupted")

        def interrupt(futures):
            running.wait(timeout=3)
            raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(audit, "run_probe", side_effect=probe),
                patch.object(audit, "as_completed", side_effect=interrupt),
            ):
                self.assertFalse(
                    audit.run_models(report, Path(directory), 10, workers=2)
                )
            saved = audit.AuditReport.model_validate_json(
                (Path(directory) / "report.json").read_text()
            )
        self.assertEqual(len(started), 2)
        self.assertEqual(
            [row.run.error_code for row in saved.models[:2]], ["RunInterrupted"] * 2
        )
        self.assertTrue(all(row.run.status == "not_tested" for row in saved.models[2:]))

    def test_cancellation_stops_a_waiting_subprocess(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("probe", 0.2)
        cancel = Mock()
        cancel.is_set.side_effect = [False, True]
        with patch.object(audit, "stop_worker") as stop:
            self.assertFalse(audit.wait_for_worker(process, 10, cancel))
        stop.assert_called_once_with(process)

    def test_completed_subprocess_is_preserved_when_cancellation_arrives(self):
        process = Mock()
        process.poll.return_value = 0
        cancel = Event()
        cancel.set()
        with patch.object(audit, "stop_worker") as stop:
            self.assertTrue(audit.wait_for_worker(process, 10, cancel))
        stop.assert_not_called()

    def test_cancelled_before_start_never_launches_a_subprocess(self):
        cancel = Event()
        cancel.set()
        with tempfile.TemporaryDirectory() as directory:
            job = audit.ProbeJob(
                model_id="test",
                profile="test",
                region="us-east-1",
                query="test",
                artifact_dir=Path(directory),
            )
            with patch.object(audit.subprocess, "Popen") as launch:
                result = audit.run_probe(job, 10, cancel)
            launch.assert_not_called()
        self.assertEqual(result.status, "not_tested")

    def test_rich_summary_contains_statuses_and_literal_error_text(self):
        report = audit.AuditReport(
            profile="test", region="us-east-1", models=[target("good"), target("bad")]
        )
        report.models[0].run = audit.RunResult(
            status="passed", reason="Completed", seconds=1.2
        )
        report.models[1].run = audit.RunResult(
            status="failed", reason="[red]literal API error[/red]"
        )
        output = io.StringIO()
        console = Console(
            file=output, force_terminal=True, color_system="standard", width=150
        )
        with patch.object(audit, "CONSOLE", console):
            audit.show_summary(report, Path("/tmp/reports"))
        text = output.getvalue()
        self.assertIn("Bedrock / SPARQ results", text)
        self.assertIn("PASSED", text)
        self.assertIn("FAILED", text)
        self.assertIn("[red]literal API error[/red]", text)
        self.assertIn("\x1b[", text)


class TestWorkerOptions(unittest.TestCase):
    def test_cli_defaults_to_four_workers_and_accepts_override(self):
        session = Mock(region_name="us-east-1")
        session.client.return_value.list_foundation_models.return_value = {
            "modelSummaries": [foundation("text")]
        }
        session.client.return_value.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": []
        }
        for arguments, expected in [
            ([], 4),
            (["--workers", "7"], 7),
            (["--mode", "pipeline"], 4),
        ]:
            with (
                self.subTest(workers=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                with (
                    patch("sparq.settings.ENVSettings"),
                    patch.object(audit.boto3, "Session", return_value=session),
                    patch.object(audit, "run_models", return_value=True) as run,
                    patch.object(audit, "CONSOLE", Console(file=io.StringIO())),
                ):
                    audit.main(
                        [
                            "--cache-dir",
                            str(Path(directory) / "cache"),
                            "--output",
                            directory,
                            *arguments,
                        ]
                    )
                self.assertEqual(run.call_args.kwargs["workers"], expected)
                report = run.call_args.args[0]
                mode = "pipeline" if "pipeline" in arguments else "capabilities"
                self.assertEqual(report.mode, mode)
                self.assertEqual(
                    report.query,
                    audit.QUERY if mode == "pipeline" else audit.CAPABILITY_QUERY,
                )

    def test_zero_workers_is_rejected_before_aws_session_creation(self):
        with (
            patch.object(audit.boto3, "Session") as session,
            patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            audit.main(["--workers", "0"])
        self.assertEqual(raised.exception.code, 2)
        session.assert_not_called()

    def test_real_parallel_subprocesses_stop_when_cancelled(self):
        cancel = Event()
        launched = Barrier(3)
        processes = []
        lock = Lock()
        real_popen = subprocess.Popen

        def launch_sleeping_worker(*args, **kwargs):
            import sys

            process = real_popen(
                [sys.executable, "-c", "import time; time.sleep(60)"], **kwargs
            )
            with lock:
                processes.append(process)
            launched.wait(timeout=3)
            return process

        try:
            with (
                tempfile.TemporaryDirectory() as directory,
                patch.object(
                    audit.subprocess, "Popen", side_effect=launch_sleeping_worker
                ),
                audit.ThreadPoolExecutor(max_workers=2) as pool,
            ):
                jobs = [
                    audit.ProbeJob(
                        model_id=str(index),
                        profile="test",
                        region="us-east-1",
                        query="test",
                        artifact_dir=Path(directory) / str(index),
                    )
                    for index in range(2)
                ]
                futures = [
                    pool.submit(audit.run_probe, job, 30, cancel) for job in jobs
                ]
                try:
                    launched.wait(timeout=3)
                finally:
                    cancel.set()
                results = [future.result(timeout=5) for future in futures]
            self.assertEqual(
                [result.error_code for result in results], ["RunInterrupted"] * 2
            )
            self.assertTrue(all(process.poll() is not None for process in processes))
        finally:
            cancel.set()
            for process in processes:
                audit.stop_worker(process)


class TestProfileOnlyOnDemand(unittest.TestCase):
    def test_sonnet_profile_only_catalog_entry_allows_us_and_global_profiles(self):
        base_id = "anthropic.claude-sonnet-4-5-20250929-v1:0"
        sonnet = foundation(base_id)
        sonnet["inputModalities"] = ["TEXT", "IMAGE"]
        sonnet["inferenceTypesSupported"] = ["INFERENCE_PROFILE"]
        client = Mock()
        client.list_foundation_models.return_value = {"modelSummaries": [sonnet]}
        client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": prefix + base_id,
                    "models": [
                        {
                            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/"
                            + base_id
                        }
                    ],
                }
                for prefix in ["us.", "global."]
            ],
        }
        models, errors = audit.discover_models(client)
        self.assertEqual(errors, [])
        by_id = {model.model_id: model for model in models}
        self.assertIn(
            "inference-profile IDs", audit.exclusion_reason(by_id[base_id]) or ""
        )
        self.assertIsNone(audit.exclusion_reason(by_id["us." + base_id]))
        self.assertIsNone(audit.exclusion_reason(by_id["global." + base_id]))
        report = audit.AuditReport(profile="test", region="us-east-1", models=models)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                audit,
                "run_probe",
                side_effect=lambda *args: audit.RunResult(
                    status="passed", reason="Completed"
                ),
            ) as probe:
                audit.run_models(report, Path(directory), 10)
            self.assertEqual(
                {call.args[0].model_id for call in probe.call_args_list},
                {"us." + base_id, "global." + base_id},
            )
            saved = audit.AuditReport.model_validate_json(
                (Path(directory) / "report.json").read_text()
            )
        self.assertEqual(
            next(row for row in saved.models if row.model_id == base_id).run.status,
            "skipped",
        )
        self.assertEqual(sum(row.run.status == "passed" for row in saved.models), 2)

    def test_profile_with_mixed_direct_and_profile_only_destinations_is_eligible(self):
        direct = foundation("direct")
        profile_only = foundation("profile-only")
        profile_only["inferenceTypesSupported"] = ["INFERENCE_PROFILE"]
        client = Mock()
        client.list_foundation_models.return_value = {
            "modelSummaries": [direct, profile_only]
        }
        client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "mixed-profile",
                    "models": [
                        {"modelArn": "arn:foundation-model/direct"},
                        {"modelArn": "arn:foundation-model/profile-only"},
                    ],
                }
            ],
        }
        models, _ = audit.discover_models(client)
        profile = next(row for row in models if row.model_id == "mixed-profile")
        self.assertIsNone(audit.exclusion_reason(profile))

    def test_provisioned_only_profile_is_still_excluded(self):
        profile = target("provisioned").model_copy(
            update={"kind": "inference_profile", "inference_types": ["PROVISIONED"]}
        )
        self.assertIn("ON_DEMAND", audit.exclusion_reason(profile) or "")
