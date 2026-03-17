from functools import partial
from typing import Optional

from sparq.settings import Settings
from sparq.nodes.planner import planner_node
from sparq.nodes.executor import executor_node
from sparq.nodes.router import router_func, router_node
from sparq.nodes.aggregator import aggregator_node
from sparq.nodes.saver import saver_node
from sparq.schemas.state import State
from sparq.utils.get_llm import get_llm

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
import pydantic_core
from rich import print

class Agentic_system:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings() # Use default settings if none provided
        self.llm_config = self.settings.LLM_CONFIG
        self.llms = self._get_llms()
        self.prompts = self.settings.load_prompts()

    def _get_llms(self):
        router_config = self.llm_config['router']
        planner_config = self.llm_config['planner']
        executor_config = self.llm_config['executor']
        aggregator_config = self.llm_config['aggregator']

        return {
            'router_llm': get_llm(model=router_config['model'], provider=router_config['provider']),
            'planner_llm': get_llm(model=planner_config['model'], provider=planner_config['provider']),
            'executor_llm': get_llm(model=executor_config['model'], provider=executor_config['provider']),
            'aggregator_llm': get_llm(model=aggregator_config['model'], provider=aggregator_config['provider'])
        }
    
    def _get_node_definitions(self):
        self.router_node_partial = partial(router_node, llm=self.llms['router_llm'], prompt=self.prompts['router_prompt'])
        self.planner_node_partial = partial(planner_node, llm=self.llms['planner_llm'], sys_prompt=self.prompts['planner_prompt'], settings=self.settings)
        self.executor_node_partial = partial(executor_node, llm=self.llms['executor_llm'], prompt=self.prompts['executor_prompt'], output_dir=self.settings.EXECUTOR_OUTPUT_DIR)
        self.aggregator_node_partial = partial(aggregator_node, llm=self.llms['aggregator_llm'], prompt=self.prompts['aggregator_prompt'])
        self.saver_node_partial = partial(saver_node, save_dir=self.settings.OUTPUT_DIR)
    
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
        async for chunk in self.graph.astream(input=input_data, stream_mode="updates", config={'recursion_limit': 100}):
            print(chunk)

    def save_results(self):
        pass
