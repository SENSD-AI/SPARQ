from functools import partial

from pathlib import Path

# from sparq.settings_old import Settings
from sparq.settings import AgenticSystemSettings
from sparq.nodes.planner import planner_node
from sparq.nodes.executor import executor_node
from sparq.nodes.router import router_func, router_node
from sparq.nodes.aggregator import aggregator_node
from sparq.nodes.saver import saver_node
from sparq.schemas.state import State
from sparq.utils.get_package_dir import get_package_dir
from sparq.utils.helpers import load_text

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
import pydantic_core
from rich import print

class Agentic_system:
    def __init__(self):
        self.settings = AgenticSystemSettings()         # Ignore 'missing parameters' warning.

        # Get system prompts
        self.prompts_dir = self.settings.paths.prompts_dir
        self.system_prompts = self._load_prompts(self.prompts_dir)

    def _load_prompts(self, prompts_dir: Path) -> dict:

        # WHY do we need is_absolute() check?
        if not prompts_dir.is_absolute():
            prompts_dir = get_package_dir() / prompts_dir
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
        self.executor_node_partial = partial(executor_node, llm_config=llm_config.executor, prompt=self.system_prompts['executor_prompt'], output_dir=run_dir / "executor")
        self.aggregator_node_partial = partial(aggregator_node, llm_config=llm_config.aggregator, prompt=self.system_prompts['aggregator_prompt'])
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
        graph_init.add_conditional_edges("router", router_func, {True: "planner", False: END})
        graph_init.add_edge("planner", "executor")
        graph_init.add_edge("executor", "aggregator")
        graph_init.add_edge("aggregator", "saver")
        graph_init.add_edge("saver", END)
        
        self.graph = graph_init.compile()

    async def run(self, user_query: str):
        self._get_node_definitions()
        self._build_graph()

        input_data = {"query": user_query} # This will go into the State schema expected by the graph
        async for chunk in self.graph.astream(input=input_data, stream_mode="updates"):
            print(chunk)

    def save_results(self):
        pass
