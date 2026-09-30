import uuid
from functools import partial
from pathlib import Path
import datetime

import pydantic_core
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from sparq.architectures.v1.nodes.aggregator import aggregator_node
from sparq.architectures.v1.nodes.executor import executor_node
from sparq.architectures.v1.nodes.planner import planner_node
from sparq.architectures.v1.nodes.router import router_func, router_node
from sparq.architectures.v1.nodes.saver import saver_node

# from sparq.settings_old import Settings
from sparq.architectures.v1.settings import V1Settings
from sparq.logging_config import logger, run_log_context
from sparq.schemas.output_schemas import EvaluationContext, SystemOutput
from sparq.schemas.state import State
from sparq.tools.python_repl.namespace import cleanup_run
from sparq.utils.helpers import load_text, write_json


class Agentic_system:
    def __init__(self, verbose: bool = False, *, settings: V1Settings | None = None):
        self.settings = settings if settings is not None else V1Settings(verbose=verbose)

        # Get system prompts
        self.prompts_dir = self.settings.paths.prompts_dir
        self.system_prompts = self._load_prompts(self.prompts_dir)

    def _load_prompts(self, prompts_dir: Path) -> dict:
        return {
            'router_prompt':     load_text(prompts_dir / "router_message.txt"),
            'planner_prompt':    load_text(prompts_dir / "planner_message.txt"),
            'executor_prompt':   load_text(prompts_dir / "executor_message.txt"),
            'aggregator_prompt': load_text(prompts_dir / "aggregator_message.txt"),
        }

    def _get_node_definitions(self):
        llm_config = self.settings.llm_config
        run_dir = self.settings.paths.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.router_node_partial = partial(router_node, llm_config=llm_config.router, prompt=self.system_prompts['router_prompt'])
        self.planner_node_partial = partial(planner_node, llm_config=llm_config.planner, sys_prompt=self.system_prompts['planner_prompt'])
        self.executor_node_partial = partial(executor_node, llm_config=llm_config.executor, worker_prompt=self.system_prompts['executor_prompt'], output_dir=run_dir / "executor")
        self.aggregator_node_partial = partial(aggregator_node, llm_config=llm_config.aggregator, prompt=self.system_prompts['aggregator_prompt'], working_dir=run_dir / "executor")
        self.saver_node_partial = partial(saver_node, save_dir=run_dir)
    
    def _build_graph(self):
        graph_init = StateGraph(state_schema=State)
        graph_init.add_node("router", self.router_node_partial)
        graph_init.add_node(
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

    async def run(
        self,
        user_query: str,
        run_id: str | None = None,
        difficulty: int = 0,
        evaluation_context: EvaluationContext | None = None,
    ):
        self._get_node_definitions()
        self._build_graph()

        # Generate a run ID if not given
        if not run_id:
            run_id = str(uuid.uuid4())

        run_dir = self.settings.paths.run_dir
        assert run_dir is not None

        input_data = {"query": user_query} # This will go into the State schema expected by the graph
        with run_log_context(run_dir, run_id):
            try:
                t0 = datetime.datetime.now()
                async for state_values in self.graph.astream(input=input_data,
                                                    config={"configurable": {"run_id": run_id}},
                                                    stream_mode="values"):
                    logger.debug(state_values)
                    write_json(run_dir, 'trace.json', state_values)
            finally:
                cleanup_run(run_id)
                tf = datetime.datetime.now()
                duration = (tf - t0).total_seconds()

        result = SystemOutput(
            run_id=run_id,
            query=user_query,
            difficulty=difficulty,
            models=self.settings.llm_config.model_dump(),
            response=state_values['answer'],
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

        write_json(run_dir, 'result.json', result)

        return result
