"""Verify name/profile mappings and explicit model selection without inference."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rich.console import Console

from tests.utils.bedrock_utils.test_audit import foundation
from utils.bedrock_utils import audit

BASE = "anthropic.example-v1:0"
NAME = "Example Model"
APP_ARN = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/app-1"


def catalog_client() -> Mock:
    model = foundation(BASE)
    model.update(modelName=NAME, inferenceTypesSupported=["INFERENCE_PROFILE"])
    other = foundation("other-model")
    client = Mock()
    client.list_foundation_models.return_value = {"modelSummaries": [model, other]}
    client.list_inference_profiles.side_effect = [
        {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "us." + BASE,
                    "inferenceProfileName": "US Example",
                    "type": "SYSTEM_DEFINED",
                    "status": "ACTIVE",
                    "inferenceProfileArn": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us."
                    + BASE,
                    "models": [
                        {
                            "modelArn": f"arn:aws:bedrock:{region}::foundation-model/{BASE}"
                        }
                        for region in ("us-east-1", "us-west-2")
                    ],
                },
            ],
            "nextToken": "next",
        },
        {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "global." + BASE,
                    "inferenceProfileName": "Global Example",
                    "type": "SYSTEM_DEFINED",
                    "status": "ACTIVE",
                    "models": [
                        {
                            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/"
                            + BASE
                        }
                    ],
                },
                {
                    "inferenceProfileId": "app-1",
                    "inferenceProfileName": "Application Example",
                    "inferenceProfileArn": APP_ARN,
                    "type": "APPLICATION",
                    "status": "ACTIVE",
                    "models": [
                        {
                            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/"
                            + BASE
                        }
                    ],
                },
                {
                    "inferenceProfileId": "us.other-model",
                    "inferenceProfileName": "Other Profile",
                    "models": [
                        {
                            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/other-model"
                        }
                    ],
                },
            ]
        },
    ]
    return client


class TestProfileMapping(unittest.TestCase):
    def test_multiple_routes_and_application_id_are_retained(self):
        models, errors = audit.discover_models(catalog_client())
        mapping = audit.build_profile_map(models, "test", "us-east-1", errors)
        expected = sorted(["us." + BASE, "global." + BASE, "app-1"])
        self.assertEqual(mapping.by_name[NAME], expected)
        self.assertEqual(mapping.by_model_id[BASE], expected)
        self.assertEqual(mapping.by_name["US Example"], ["us." + BASE])
        self.assertEqual(mapping.profiles["app-1"].invocation_id, APP_ARN)
        self.assertEqual(mapping.profiles["app-1"].type, "APPLICATION")
        self.assertEqual(mapping.profiles["app-1"].status, "ACTIVE")
        self.assertEqual(mapping.profiles["us." + BASE].foundation_model_ids, [BASE])
        base = next(m for m in models if m.model_id == BASE)
        self.assertEqual(base.inference_profile_ids, expected)
        self.assertIn("us." + BASE, audit.exclusion_reason(base) or "")

    def test_foundation_names_expand_but_explicit_profile_ids_do_not(self):
        models, _ = audit.discover_models(catalog_client())
        for selector in (BASE, NAME):
            selected, unknown = audit.select_models(models, [selector])
            self.assertEqual(unknown, [])
            self.assertEqual(
                {m.model_id for m in selected},
                {"us." + BASE, "global." + BASE, APP_ARN},
            )
        for selector, expected in [
            ("US Example", "us." + BASE),
            ("global." + BASE, "global." + BASE),
            ("app-1", APP_ARN),
        ]:
            selected, unknown = audit.select_models(models, [selector])
            self.assertEqual(unknown, [])
            self.assertEqual([m.model_id for m in selected], [expected])
        selected, unknown = audit.select_models(models, ["missing"])
        self.assertEqual(selected, [])
        self.assertEqual(unknown, ["missing"])

    def test_partial_discovery_is_explicit_in_saved_mapping(self):
        models, _ = audit.discover_models(catalog_client())
        mapping = audit.build_profile_map(
            models, "test", "us-east-1", ["Second page denied"]
        )
        self.assertFalse(mapping.complete)
        self.assertEqual(mapping.discovery_errors, ["Second page denied"])
        self.assertIn(BASE, mapping.by_model_id)

    def test_targeted_list_only_saves_the_complete_map_without_inference(self):
        session = Mock(region_name="us-east-1")
        session.client.return_value = catalog_client()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("sparq.settings.ENVSettings"),
                patch.object(audit.boto3, "Session", return_value=session),
                patch.object(audit, "CONSOLE", Console(file=io.StringIO())),
                patch.object(audit, "run_models") as run,
            ):
                code = audit.main(
                    [
                        "--cache-dir",
                        str(Path(directory) / "cache"),
                        "--profile",
                        "test",
                        "--region",
                        "us-east-1",
                        "--list-only",
                        "--model-id",
                        NAME,
                        "--output",
                        directory,
                    ]
                )
            self.assertEqual(code, 0)
            run.assert_not_called()
            report = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(len(report["models"]), 3)
            mapping = json.loads(
                (Path(directory) / "inference_profiles.json").read_text()
            )
            self.assertTrue(mapping["complete"])
            self.assertEqual(mapping["by_model_id"]["other-model"], ["us.other-model"])
            self.assertEqual(mapping["region"], "us-east-1")
