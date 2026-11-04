import datetime
import uuid
from functools import partial
from pathlib import Path
from typing import Any
import json

import pydantic_core
from pydantic_core import to_jsonable_python
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from sparq.architectures.v1.nodes.aggregator import aggregator_node
from sparq.architectures.v1.nodes.executor import executor_node
from sparq.architectures.v1.nodes.planner import planner_node
from sparq.architectures.v1.nodes.router import router_func, router_node
from sparq.architectures.v1.nodes.saver import saver_node
from sparq.architectures.v1.settings import V1Settings
from sparq.logging_config import run_log_context
from sparq.schemas.runs import EvaluationContext, SystemOutput
from sparq.schemas.state import State
from sparq.tools.python_repl.namespace import cleanup_run
from sparq.utils.get_llm import get_llm
from sparq.utils.helpers import load_text, write_json


class Agentic_system:
    def __init__(self, *, settings: V1Settings | None = None):
        self.settings = settings if settings is not None else V1Settings()

        # Get system prompts
        self.prompts_dir = self.settings.paths.prompts_dir
        self.system_prompts = self._load_prompts(self.prompts_dir)
        run_dir = self.settings.paths.run_dir
        assert run_dir is not None
        self.run_dir = run_dir

    def _load_prompts(self, prompts_dir: Path) -> dict:
        return {
            'router_prompt':     load_text(prompts_dir / "router_message.txt"),
            'planner_prompt':    load_text(prompts_dir / "planner_message.txt"),
            'executor_prompt':   load_text(prompts_dir / "executor_message.txt"),
            'aggregator_prompt': load_text(prompts_dir / "aggregator_message.txt"),
        }

    def _get_node_definitions(self):
        llm_config = self.settings.llm_config
        run_dir = self.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        vision_model: BaseChatModel = get_llm(
            model=llm_config.vision_model.model_name,
            provider=llm_config.vision_model.provider,
            max_tokens=llm_config.vision_model.max_tokens,
            temperature=llm_config.vision_model.temperature,
            top_p=llm_config.vision_model.top_p,
            top_k=llm_config.vision_model.top_k,
        )
        self.router_node_partial = partial(router_node, llm_config=llm_config.router, prompt=self.system_prompts['router_prompt'])
        self.planner_node_partial = partial(planner_node, llm_config=llm_config.planner, sys_prompt=self.system_prompts['planner_prompt'])
        self.executor_node_partial = partial(
            executor_node,
            llm_config=llm_config.executor,
            worker_prompt=self.system_prompts['executor_prompt'],
            output_dir=run_dir / "executor",
            vision_model=vision_model,
        )
        self.aggregator_node_partial = partial(
            aggregator_node,
            llm_config=llm_config.aggregator,
            prompt=self.system_prompts['aggregator_prompt'],
            working_dir=run_dir,
            vision_model=vision_model,
        )
        self.saver_node_partial = partial(saver_node, save_dir=run_dir, llm_config=llm_config)
    
    def _build_graph(self):
        graph_init = StateGraph(state_schema=State)
        graph_init.add_node("router", self.router_node_partial)
        graph_init.add_node( # TODO: Fix this pylance error
            "planner",
            self.planner_node_partial,
            retry=RetryPolicy(
                max_attempts=5,
                retry_on=pydantic_core._pydantic_core.ValidationError
                )
            )
        graph_init.add_node("executor", self.executor_node_partial)
        graph_init.add_node("aggregator", self.aggregator_node_partial)
        graph_init.add_node("saver", self.saver_node_partial)
        
        graph_init.add_edge(START, "router")
        graph_init.add_conditional_edges("router", router_func, {True: "planner", False: "saver"})
        graph_init.add_edge("planner", "executor")
        graph_init.add_edge("executor", "aggregator")
        graph_init.add_edge("aggregator", "saver")
        graph_init.add_edge("saver", END)
        
        self.graph = graph_init.compile()

    def _log_graph_event(self, chunk: dict[str, Any]) -> None:
        """Append one outer or nested graph event to the run's JSONL file."""
        events_path = self.run_dir / "events.jsonl"
        event = to_jsonable_python(chunk, by_alias=False)
        with events_path.open("a", encoding="utf-8") as events_file:
            events_file.write(json.dumps(event) + "\n")

    async def run(
        self,
        user_query: str,
        run_id: str | None = None,
        difficulty: int = 0,
        evaluation_context: EvaluationContext | None = None,
    ):
        question_asked_at = datetime.datetime.now(datetime.UTC)
        self._get_node_definitions()
        self._build_graph()

        # Generate a run ID if not given
        if not run_id:
            run_id = str(uuid.uuid4())

        input_data = {
            "query": user_query,
            "question_asked_at": question_asked_at,
            "messages": [HumanMessage(content=user_query)],
        }
        state_values: State | None = None
        with run_log_context(self.run_dir, run_id):
            t0 = datetime.datetime.now()
            try:
                # LangGraph 1.2.8's v2 mapper coerces nested values into the root schema.
                async for namespace, mode, data in self.graph.astream(
                    input=input_data,
                    config={"configurable": {"run_id": run_id}},
                    stream_mode=["values", "updates"],
                    subgraphs=True,
                    version="v1",
                ):
                    event = {"type": mode, "ns": namespace, "data": data}
                    self._log_graph_event(event)

                    # Nested agents have their own state; retain only the root state.
                    if mode == "values" and not namespace:
                        state_values = State.model_validate(data)
                        write_json(self.run_dir, "trace.json", state_values)
            finally:
                cleanup_run(run_id)
                tf = datetime.datetime.now()
                duration = (tf - t0).total_seconds()

        if state_values is None:
            raise ValueError("Run completed without an outer graph state.")

        final_message = state_values.messages[-1]
        if not isinstance(final_message, AIMessage):
            raise ValueError("Run completed without a final assistant response.")

        result = SystemOutput(
            run_id=run_id,
            query=user_query,
            difficulty=difficulty,
            models=self.settings.llm_config.model_dump(),
            response=final_message.text,
            time_started=t0,
            time_ended=tf,
            duration=duration, 
            ablation_config={},
            cost=None,
            token_out=None,
            evaluation_context=evaluation_context,
            sparq_judge_review=None,
            sparq_judge_score=None,
        )

        write_json(self.run_dir, 'result.json', result)

        return result
