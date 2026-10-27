"""Offline checks for image analysis through SPARQ's worker and aggregator."""

import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

# Importing the aggregator otherwise downloads tokenizer data on an empty cache.
with patch("tiktoken.get_encoding", return_value=Mock(encode=lambda text: list(text))):
    from sparq.architectures.v1.nodes import aggregator, executor
    from sparq.architectures.v1.system import Agentic_system

from sparq.architectures.v1.helpers.organizer import Index
from sparq.schemas.output_schemas import Plan, Step, StepResult
from sparq.schemas.state import State
from sparq.settings import LLMSetting
from sparq.tools.image_processing import image_analysis_tool
from sparq.tools.image_processing.image_analysis_tool import ImageReport


class TestImageAnalysisIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
            "QVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
        )
        (self.root / "plot.png").write_bytes(self.image_bytes)
        self.prompts = Agentic_system._load_prompts(
            Mock(),
            Path(executor.__file__).parents[1] / "prompts",
        )
        self.image_prompt = (
            Path(image_analysis_tool.__file__).parent / "prompts" / "v0.txt"
        ).read_text()
        self.messages = []
        self.report = ImageReport(
            summary="Synthetic vision response",
            observations=["One pixel"],
            visible_text=[],
            charts_and_tables=[],
            interpretation="Fixture image",
            answer="A pixel",
            uncertainties=[],
        )

        def inspect(messages):
            self.messages.extend(messages)
            return self.report

        self.agent_model = Mock(spec=BaseChatModel)
        self.vision_model = Mock(spec=BaseChatModel)
        self.vision_model.with_structured_output.return_value = RunnableLambda(inspect)
        self.config = LLMSetting(
            model="test-model", provider="openai", recursion_limit=25
        )
        self.result = StepResult(
            id=1, step="Inspect plot", success=True, execution_results="", misc=""
        )

    async def inspect_registered_tool(self, tools):
        image_tool = next(
            item for item in tools if getattr(item, "name", None) == "analyze_image"
        )
        output = await image_tool.ainvoke(
            {"image_path": "plot.png", "prompt": "What is visible?"}
        )
        self.assertEqual(json.loads(output), self.report.model_dump())
        self.assertEqual(self.messages[0].content, self.image_prompt)
        image_url = self.messages[1].content[1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(image_url.split(",", 1)[1]), self.image_bytes)

    async def test_bedrock_request_enforces_list_fields_with_native_schema(self):
        client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        model = ChatBedrockConverse(
            model="us.anthropic.claude-sonnet-4-5-20250929-v1:0", client=client
        )
        tool = image_analysis_tool.make_image_analysis_tool(model, image_root=self.root)
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": self.report.model_dump_json()}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "metrics": {"latencyMs": 1},
        }
        with patch.object(client, "converse", return_value=response) as converse:
            output = await tool.ainvoke({"image_path": "plot.png", "prompt": "Inspect"})
        request = converse.call_args.kwargs
        text_format = request["outputConfig"]["textFormat"]
        self.assertEqual(text_format["type"], "json_schema")
        schema = json.loads(text_format["structure"]["jsonSchema"]["schema"])
        self.assertFalse(schema["additionalProperties"])
        for field in (
            "observations",
            "visible_text",
            "charts_and_tables",
            "uncertainties",
        ):
            self.assertEqual(schema["properties"][field]["type"], "array")
        self.assertNotIn("toolConfig", request)
        self.assertEqual(json.loads(output), self.report.model_dump())

    async def test_worker_calls_registered_image_tool(self):
        def create_agent(**kwargs):
            self.assertIs(kwargs["model"], self.agent_model)

            async def invoke(*args, **invoke_kwargs):
                await self.inspect_registered_tool(kwargs["tools"])
                return {"structured_response": self.result}

            return SimpleNamespace(ainvoke=invoke)

        step = Step(
            id=1,
            step_description="Inspect plot",
            datasets=[],
            rationale="Check figure",
            task_type=["visualization"],
            dependencies=[],
        )
        state = State(query="Inspect plot", completed_plan_steps=[])
        with (
            patch.object(executor, "get_llm", return_value=self.agent_model),
            patch.object(executor, "get_ns_path", return_value="test-namespace"),
            patch.object(
                executor, "merge_namespaces_of_previous_deps", return_value=(0, "")
            ),
            patch.object(executor, "make_load_dataset_tool"),
            patch.object(executor, "make_python_repl_tool"),
            patch.object(executor, "filesystemtools", return_value=[]),
            patch.object(executor, "create_agent", side_effect=create_agent),
            patch.object(executor, "organize", return_value=[]),
        ):
            output = await executor.executor_node(
                state=state.model_copy(
                    update={"plan": Plan(steps=[step], wants=None, misc=None)}
                ),
                config={"configurable": {"run_id": "test-run"}},
                llm_config=self.config,
                worker_prompt=self.prompts["executor_prompt"],
                output_dir=self.root,
                vision_model=self.vision_model,
            )
        self.assertTrue(output["results"][0].success)
        self.assertTrue(self.messages, "Worker must reach the image model")
        self.agent_model.with_structured_output.assert_not_called()

    async def test_aggregator_calls_registered_async_image_tool(self):
        final_message = AIMessage(content="Image-informed report", id="report-1")

        def create_agent(**kwargs):
            self.assertIs(kwargs["model"], self.agent_model)

            async def invoke(*args, **invoke_kwargs):
                await self.inspect_registered_tool(kwargs["tools"])
                return {"messages": [
                    HumanMessage(content="Write the final report now."),
                    AIMessage(content="", tool_calls=[{
                        "name": "analyze_image", "args": {}, "id": "image-1",
                    }]),
                    ToolMessage(content="Image evidence", tool_call_id="image-1"),
                    final_message,
                ]}

            return SimpleNamespace(ainvoke=invoke)

        with (
            patch.object(aggregator, "get_llm", return_value=self.agent_model),
            patch.object(aggregator, "count_tokens", return_value=1),
            patch.object(aggregator, "filesystemtools", return_value=[]),
            patch.object(aggregator, "create_agent", side_effect=create_agent),
        ):
            output = await aggregator.aggregator_node(
                state=State(
                    query="Inspect plot", results=[self.result], completed_plan_steps=[]
                ),
                llm_config=self.config,
                prompt=self.prompts["aggregator_prompt"],
                working_dir=self.root,
                vision_model=self.vision_model,
            )
        self.assertEqual(output, {"messages": [final_message]})
        self.assertIs(output["messages"][0], final_message)
        self.agent_model.with_structured_output.assert_not_called()

    async def test_aggregator_records_no_results_as_assistant_message(self):
        output = await aggregator.aggregator_node(
            state=State(query="Inspect plot"),
            llm_config=self.config,
            prompt=self.prompts["aggregator_prompt"],
            working_dir=self.root,
            vision_model=self.vision_model,
        )

        self.assertEqual(set(output), {"messages"})
        self.assertIsInstance(output["messages"][0], AIMessage)
        self.assertIn("There are no results", output["messages"][0].text)

    async def test_aggregator_tools_read_executor_index_from_run_root(self):
        executor_dir = self.root / "executor"
        figure = executor_dir / "figures/plot.png"
        figure.parent.mkdir(parents=True)
        figure.write_bytes(self.image_bytes)
        (executor_dir / "index.json").write_text(
            Index(figures={"f1": Path("figures/plot.png")}).model_dump_json()
        )

        def create_agent(**kwargs):
            async def invoke(*args, **invoke_kwargs):
                tools = {item.name: item for item in kwargs["tools"]}
                paths = await tools["list_artifacts"].ainvoke({})
                self.assertEqual(paths, ["executor/figures/plot.png"])
                matches = await tools["search_artifact"].ainvoke(
                    {"artifact_name": paths[0]}
                )
                self.assertEqual(matches, paths)
                self.assertTrue((self.root / paths[0]).is_file())
                return {"messages": [AIMessage(content="Indexed report")]}

            return SimpleNamespace(ainvoke=invoke)

        with (
            patch.object(aggregator, "get_llm", return_value=self.agent_model),
            patch.object(aggregator, "count_tokens", return_value=1),
            patch.object(aggregator, "filesystemtools", return_value=[]),
            patch.object(aggregator, "create_agent", side_effect=create_agent),
        ):
            output = await aggregator.aggregator_node(
                state=State(
                    query="Inspect plot", results=[self.result], completed_plan_steps=[]
                ),
                llm_config=self.config,
                prompt=self.prompts["aggregator_prompt"],
                working_dir=self.root,
                vision_model=self.vision_model,
            )
        self.assertEqual(output["messages"][0].text, "Indexed report")

    async def test_system_builds_graph_without_loading_image_prompt(self):
        system = Mock(spec=Agentic_system)
        system.system_prompts = self.prompts
        system.settings = SimpleNamespace(
            paths=SimpleNamespace(run_dir=self.root),
            llm_config=SimpleNamespace(
                router=self.config,
                planner=self.config,
                executor=self.config,
                aggregator=self.config,
                vision_model=self.config.model_copy(
                    update={"model_name": "vision-model"}
                ),
            ),
        )
        with patch(
            "sparq.architectures.v1.system.get_llm", return_value=self.vision_model
        ) as get_vision_model:
            Agentic_system._get_node_definitions(system)
        get_vision_model.assert_called_once_with(
            model="vision-model", provider="openai"
        )
        Agentic_system._build_graph(system)
        self.assertIn("aggregator", system.graph.nodes)
        self.assertNotIn("image_analysis_prompt", system.system_prompts)
        for node in (system.executor_node_partial, system.aggregator_node_partial):
            self.assertNotIn("image_analysis_prompt", node.keywords)
            self.assertIs(node.keywords["vision_model"], self.vision_model)
            self.assertIs(node.keywords["llm_config"], self.config)
