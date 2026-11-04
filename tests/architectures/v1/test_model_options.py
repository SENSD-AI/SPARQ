"""Check generation options at node-to-model construction boundaries."""

import unittest
from unittest.mock import Mock, patch

from sparq.architectures.v1.nodes import planner, router
from sparq.schemas.state import State
from sparq.settings import LLMSetting


class TestNodeModelOptions(unittest.TestCase):
    def test_planner_and_router_forward_explicit_generation_arguments(self):
        for configured in (False, True):
            options = (
                {"max_tokens": 8192, "temperature": 0, "top_p": 0.9, "top_k": 40}
                if configured
                else {}
            )
            config = LLMSetting(
                model="example",
                provider="aws_bedrock",
                recursion_limit=25,
                effort="high",
                **options,
            )
            for module in (planner, router):
                with (
                    self.subTest(node=module.__name__, configured=configured),
                    patch.object(module, "get_llm", return_value=Mock()) as create,
                ):
                    state = State(query="Question", completed_plan_steps=[])
                    if module is planner:
                        with patch.object(
                            planner, "load_data_context", return_value="Context"
                        ):
                            planner.planner_node(state, config, "Plan {data_context}")
                    else:
                        router.router_node(state, config, "Route")
                    create.assert_called_once_with(
                        model="example",
                        provider="aws_bedrock",
                        max_tokens=options.get("max_tokens"),
                        temperature=options.get("temperature"),
                        top_p=options.get("top_p"),
                        top_k=options.get("top_k"),
                    )
