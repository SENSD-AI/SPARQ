"""Exercise catalog coverage, per-model errors and checkpointing without AWS calls."""

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from sparq.evaluation import bedrock_audit as audit


def foundation(identifier: str, output: str = "TEXT") -> dict:
    return {
        "modelId": identifier,
        "modelName": identifier,
        "inputModalities": ["TEXT"],
        "outputModalities": [output],
    }


def target(identifier: str) -> audit.ModelResult:
    return audit.ModelResult(model_id=identifier, name=identifier, kind="foundation")


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

    def test_known_non_text_models_are_explained_and_unknowns_are_attempted(self):
        model = target("embed")
        model.output_modalities = ["EMBEDDING"]
        reason = audit.exclusion_reason(model)
        assert reason is not None
        self.assertIn("TEXT output", reason)
        self.assertIsNone(audit.exclusion_reason(target("unknown-profile")))

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
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(
                audit,
                "run_probe",
                side_effect=[
                    audit.RunResult(status="passed", reason="Completed"),
                    KeyboardInterrupt(),
                ],
            ):
                self.assertFalse(audit.run_models(report, output, 10))
            saved = json.loads((output / "report.json").read_text())
            self.assertEqual(saved["models"][0]["run"]["status"], "passed")
            self.assertEqual(saved["models"][1]["run"]["error_code"], "RunInterrupted")
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
            self.assertEqual(result.status, "failed")
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
            with patch.object(audit.subprocess, "Popen", return_value=process):
                result = audit.run_probe(job, 10)
            self.assertEqual(result.error_code, "WorkerFailure")
            self.assertEqual(result.status, "failed")

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
                code = audit.main(["--list-only", "--output", directory])
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
                    side_effect=[
                        audit.RunResult(status="failed", reason="Unsupported tools"),
                        audit.RunResult(status="passed", reason="Completed"),
                    ],
                ) as probe,
            ):
                self.assertEqual(audit.main(["--output", directory]), 0)
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
