"""Probe model interfaces locally, including an actual LangChain agent/tool loop."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

from sparq.schemas.planning import Plan, Router, Step
from utils.bedrock_utils import capabilities as probes
from utils.bedrock_utils.audit import ProbeJob


class LocalModel(BaseChatModel):
    """A deterministic model that actually requests and consumes the local tool."""

    @property
    def _llm_type(self):
        return "local-probe-test"

    def bind_tools(self, tools, **kwargs):
        return self.bind(
            tools=[convert_to_openai_tool(item) for item in tools], **kwargs
        )

    def with_structured_output(self, schema, **kwargs):
        if schema is Router:
            return RunnableLambda(lambda _: Router(route=True, answer=None))
        plan = Plan(
            wants=None,
            misc=None,
            steps=[
                Step(
                    id=1,
                    step_description="Look up",
                    datasets=[],
                    rationale="Retrieve",
                    task_type=["lookup"],
                    dependencies=[],
                ),
                Step(
                    id=2,
                    step_description="Report",
                    datasets=[],
                    rationale="Answer",
                    task_type=["report"],
                    dependencies=[1],
                ),
            ],
        )
        return RunnableLambda(lambda _: plan)

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs
    ):
        outputs = [
            m
            for m in messages
            if isinstance(m, ToolMessage) and m.name == "lookup_probe_value"
        ]
        if not outputs:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_probe_value",
                        "args": {"key": "probe"},
                        "id": "lookup",
                    }
                ],
            )
        elif any(
            isinstance(t, dict) and t.get("function", {}).get("name") == "StepResult"
            for t in kwargs.get("tools", [])
        ):
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "StepResult",
                        "id": "structured",
                        "args": {
                            "id": 1,
                            "step": "Look up",
                            "success": True,
                            "execution_results": str(outputs[-1].content),
                            "files_generated": [],
                        },
                    }
                ],
            )
        else:
            message = AIMessage(content=str(outputs[-1].content))
        return ChatResult(generations=[ChatGeneration(message=message)])


class TestCapabilities(unittest.IsolatedAsyncioTestCase):
    async def test_bedrock_adapter_serializes_and_parses_all_four_probes(self):
        from langchain_aws import ChatBedrockConverse

        def converse(**request):
            names = {
                item["toolSpec"]["name"] for item in request["toolConfig"]["tools"]
            }
            values = [
                block["toolResult"]["content"][0]["text"]
                for message in request["messages"]
                for block in message["content"]
                if "toolResult" in block
            ]
            if names == {"Router"}:
                name, args = "Router", {"route": True}
            elif names == {"Plan"}:
                name = "Plan"
                args = {
                    "steps": [
                        {
                            "id": 1,
                            "step_description": "Look up",
                            "datasets": [],
                            "rationale": "Retrieve",
                            "task_type": ["lookup"],
                            "dependencies": [],
                        },
                        {
                            "id": 2,
                            "step_description": "Report",
                            "datasets": [],
                            "rationale": "Answer",
                            "task_type": ["report"],
                            "dependencies": [1],
                        },
                    ]
                }
            elif values and ("StepResult" in names or "outputConfig" in request):
                name, args = (
                    "StepResult",
                    {
                        "id": 1,
                        "step": "Look up",
                        "success": True,
                        "execution_results": values[-1],
                        "files_generated": [],
                    },
                )
                if "outputConfig" in request:
                    return {
                        "output": {
                            "message": {
                                "role": "assistant",
                                "content": [{"text": json.dumps(args)}],
                            }
                        },
                        "stopReason": "end_turn",
                        "usage": {
                            "inputTokens": 1,
                            "outputTokens": 1,
                            "totalTokens": 2,
                        },
                        "metrics": {"latencyMs": 1},
                    }
            elif values:
                return {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [{"text": values[-1]}],
                        }
                    },
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                    "metrics": {"latencyMs": 1},
                }
            else:
                name, args = "lookup_probe_value", {"key": "probe"}
            return {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "test-call",
                                    "name": name,
                                    "input": args,
                                }
                            }
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                "metrics": {"latencyMs": 1},
            }

        client = Mock()
        client.converse.side_effect = converse
        model = ChatBedrockConverse(
            model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            client=client,
            bedrock_client=Mock(),
            region_name="us-east-1",
            profile={"structured_output": False, "tool_calling": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            job = ProbeJob(
                model_id=model.model_id,
                profile="test",
                region="us-east-1",
                query="",
                artifact_dir=Path(directory),
            )
            with patch.object(probes, "get_llm", return_value=model):
                result = await probes.execute_job(job)
            self.assertEqual(result.status, "passed")
            self.assertGreaterEqual(client.converse.call_count, 6)

    async def test_full_small_probe_uses_real_tools_and_agent_without_aws(self):
        with tempfile.TemporaryDirectory() as directory:
            job = ProbeJob(
                model_id="test",
                profile="test",
                region="us-east-1",
                query="",
                artifact_dir=Path(directory),
            )
            # Even a negative metadata hint is not substituted for an observed test.
            model = LocalModel(profile={"tool_calling": False})
            with patch.object(probes, "get_llm", return_value=model):
                result = await probes.execute_job(job)
            self.assertEqual(result.status, "passed")
            self.assertEqual(len(result.checks), 4)
            self.assertTrue(
                all(check.status == "passed" for check in result.checks.values())
            )
            self.assertIsNone(result.capability_hints["structured_output"])
            self.assertFalse(result.capability_hints["tool_calling"])
            evidence = json.loads(
                (Path(directory) / "capability_evidence.json").read_text()
            )
            for name in ("tool_round_trip", "tools_with_structure"):
                self.assertEqual(len(evidence[name]["executed_tools"]), 1)
            self.assertNotEqual(
                evidence["tool_round_trip"]["executed_tools"][0]["value"],
                evidence["tools_with_structure"]["executed_tools"][0]["value"],
            )

    async def test_plain_answer_cannot_pass_tool_round_trip(self):
        class GuessingModel(LocalModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="guessed"))]
                )

        evidence = {}
        self.assertFalse(await probes.probe_tool(GuessingModel(), evidence))
        self.assertEqual(evidence["executed_tools"], [])

    async def test_access_denial_stops_remaining_checks(self):
        error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Denied"}},
            "Converse",
        )

        async def denied(*args):
            raise error

        with tempfile.TemporaryDirectory() as directory:
            job = ProbeJob(
                model_id="test",
                profile="test",
                region="us-east-1",
                query="",
                artifact_dir=Path(directory),
            )
            with (
                patch.object(probes, "get_llm", return_value=LocalModel()),
                patch.object(probes, "probe_router", side_effect=denied),
            ):
                result = await probes.execute_job(job)
            self.assertEqual(result.status, "access_denied")
            self.assertEqual(result.checks["structured_router"].status, "access_denied")
            self.assertEqual(result.checks["tool_round_trip"].status, "skipped")

    def test_error_categories_do_not_turn_validation_errors_into_unsupported(self):
        self.assertEqual(probes.classify_error(TimeoutError()).status, "timed_out")
        self.assertEqual(
            probes.classify_error(
                ValueError("This model does not support tool use")
            ).status,
            "unsupported",
        )
        self.assertEqual(
            probes.classify_error(ValueError("Invalid response schema")).status, "error"
        )

    async def test_unsupported_router_does_not_hide_other_capabilities(self):
        async def unsupported(*args):
            raise NotImplementedError("Structured router not supported")

        with tempfile.TemporaryDirectory() as directory:
            job = ProbeJob(
                model_id="test",
                profile="test",
                region="us-east-1",
                query="",
                artifact_dir=Path(directory),
            )
            with (
                patch.object(probes, "get_llm", return_value=LocalModel()),
                patch.object(probes, "probe_router", side_effect=unsupported),
            ):
                result = await probes.execute_job(job)
            self.assertEqual(result.status, "unsupported")
            self.assertEqual(result.checks["structured_router"].status, "unsupported")
            self.assertEqual(result.checks["tools_with_structure"].status, "passed")

    async def test_timeout_is_inconclusive_and_other_checks_continue(self):
        async def timeout(*args):
            raise TimeoutError()

        with tempfile.TemporaryDirectory() as directory:
            job = ProbeJob(
                model_id="test",
                profile="test",
                region="us-east-1",
                query="",
                artifact_dir=Path(directory),
            )
            with (
                patch.object(probes, "get_llm", return_value=LocalModel()),
                patch.object(probes, "probe_router", side_effect=timeout),
            ):
                result = await probes.execute_job(job)
            self.assertEqual(result.status, "timed_out")
            self.assertEqual(result.checks["tool_round_trip"].status, "passed")
