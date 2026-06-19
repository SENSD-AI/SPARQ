from pathlib import Path
from typing import List

from sparq.schemas.state import State, WorkerState
from sparq.schemas.output_schemas import Plan, Step, ExecutorOutput
from sparq.settings import LLMSetting
from sparq.utils import helpers
from sparq.utils.get_llm import get_llm
from sparq.tools.python_repl.python_repl_tool import make_python_repl_tool
from sparq.tools.python_repl.namespace import get_ns_path, load_ns
from sparq.tools.filesystemtools import filesystemtools
from sparq.tools.data_discovery_tools import make_load_dataset_tool, get_sheet_names, find_csv_excel_files, get_cached_dataset_path

from langgraph.graph import END
from langgraph.types import Send
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.config import RunnableConfig


def _build_context(results: dict, ns_path: str) -> str:
    """
    Build a context string for the current step based on previous results and the current namespace.

    Args:
        results: Previously completed step results.
        ns_path: Path to the run-scoped namespace pickle file used to enumerate live variables.

    What is added to the context?
    - Previously completed steps and their results
    - Notes or miscellaneous information from previous steps
    - Variables currently in the namespace and their types

    """
    lines = []

    if results != {}:
        lines.append("Previously completed steps:")
        for step, data in results.items():
            lines.append(f"\n{step}")
            lines.append(f"  Results: {data['execution_results']}")
            if data['misc']:
                lines.append(f"  Notes: {data['misc']}")

    ns = load_ns(ns_path)
    ns_vars = {k: type(v).__name__ for k, v in ns.items() if not k.startswith("__")}
    if ns_vars:
        lines.append("\nVariables currently in namespace:")
        for var, typename in ns_vars.items():
            lines.append(f"  {var}: {typename}")

    return "\n".join(lines)


def executor_node(state: State):
    """
    Execute the plan.

    Args:
        state: LangGraph state containing the plan and data context.
        config: LangGraph RunnableConfig; must contain config["configurable"]["run_id"] to scope the REPL namespace.
        llm_config: LLM model and provider settings for the executor.
        prompt: System prompt template string.
        output_dir: Directory where executor outputs (plots, files) are written.
    """
    print("Executing plan to answer your query")
    plan = state.plan
    completed = set(state.completed_steps)

    while len(completed) < len(plan.steps):
        ready_steps = []

        for step in plan.steps:
            if step.id in completed:
                continue

            # A step is ready to be executed if it has no dependencies
            deps = step.dependencies

            # Check if dependencies are completed
            if all(dep in completed for dep in deps):
                ready_steps.append(step)

        if not ready_steps:
            raise ValueError("Deadlock detected in executor node: Unresolved dependencies remain.")

        print(f"[Executor Node] Spawning {len(ready_steps)} parallel worker agents...")

        # TODO: dispatch ready_steps to worker agents and merge results back into completed/results


async def execute_single_step_worker(step: Step, llm_config: LLMSetting, prompt: str, output_dir: Path):
    """An isolated worker instance spawned for an individual plan step"""
    print(f"[Worker Agent] starting step {step.id}: {step.step_description}")

    # TODO: Complete the function
    pass




async def execute_step(state: State, config: RunnableConfig, llm_config: LLMSetting, prompt: str, output_dir: Path):
    """
    Execute the plan.

    Args:
        state: LangGraph state containing the plan and data context.
        config: LangGraph RunnableConfig; must contain config["configurable"]["run_id"] to scope the REPL namespace.
        llm_config: LLM model and provider settings for the executor.
        prompt: System prompt template string.
        output_dir: Directory where executor outputs (plots, files) are written.
    """
    print("Executing plan to answer your query.")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = config.get("configurable", {}).get("run_id", "default")
    ns_path = get_ns_path(run_id)

    plan: Plan = state.plan
    llm = get_llm(model=llm_config.model_name, provider=llm_config.provider)

    data_context = state.data_context

    _tools = [
        make_load_dataset_tool(ns_path),
        get_sheet_names,
        make_python_repl_tool(ns_path),
        find_csv_excel_files,
        get_cached_dataset_path,
    ] + (filesystemtools(working_dir=str(output_dir), selected_tools='all'))

    system_prompt_template: BasePromptTemplate = PromptTemplate.from_template(prompt).partial(
        data_context=str(data_context),
        output_dir=str(output_dir)
    )
    system_prompt_str: str = system_prompt_template.invoke(input={}).to_string()
    system_prompt: SystemMessage = SystemMessage(content=system_prompt_str)

    # create the ReAct agent
    agent = create_react_agent(
        model=llm,
        tools=_tools,
        prompt=system_prompt,
        response_format=(prompt, ExecutorOutput),
    )

    def process_step(results: dict, step_description, step_index):
        context = _build_context(results, ns_path)
        user_content = f"{context}\n\nYour current task:\n{step_description}" if context else step_description
        agent_input = {"messages": [{"role": "user", "content": user_content}]}
        response = agent.invoke(agent_input, config={"recursion_limit": llm_config.recursion_limit})
        structured_response = response["structured_response"]

        outer_dict_key = f"Step {step_index}: {step_description}"
        results[outer_dict_key] = {
            'execution_results': structured_response.execution_results,
            'files_generated': structured_response.files_generated,
            'misc': structured_response.misc,
        }

        return results

    results = {}
    for i, step in enumerate(plan.steps):
        try:
            results = process_step(results, step.step_description, i+1)
        except Exception as e:
            results[f"Step {i+1}: {step.step_description}"] = {"error": str(e)}
        

    return {'executor_results': results}

def test_executor(plan: Plan):
    print(f"Testing executor node with sample plan: \n {plan.pretty_print()}")

    from sparq.settings import (
        ENVSettings,
        PathSettings,
        DATA_MANIFEST_PATH,
        DATA_SUMMARIES_SHORT_PATH,
    )
    from sparq.architectures.v1.settings import V1Settings
    from sparq.schemas.data_context import load_data_context

    # Load environment and system settings
    env_settings = ENVSettings()
    system_settings = V1Settings()

    # Get system prompt
    prompt = helpers.load_text(system_settings.paths.prompts_dir / "executor_message.txt")
    updated_paths = PathSettings.model_validate(
        system_settings.paths.model_dump() | {"output_stem": "test_executor"}
    )
    run_dir = updated_paths.run_dir

    data_context = load_data_context(DATA_MANIFEST_PATH, DATA_SUMMARIES_SHORT_PATH)

    assert run_dir is not None
    state = State(query="", route=None, answer=None, plan=plan, data_context=data_context, executor_results={})
    response = executor_node(state=state, config={"configurable": {"run_id": "test"}}, llm_config=system_settings.llm_config.executor, prompt=prompt, output_dir=run_dir)
    
    for step, result in response['executor_results'].items():
        print(step)
        print(result)

if __name__ == "__main__":
    from sparq.schemas.output_schemas import Step

    sample_plan = Plan(
        steps=[
            Step(
                id=1,
                step_description="Get the pulsenet dataset and load it into a dataframe.",
                datasets=[],
                rationale="The pulsenet dataset contains information about various pathogens, including their serotypes and sources of isolation. Loading it into a dataframe will allow us to analyze the data and find correlations.",
                task_type=["data_mining"],
                dependencies=None
            ),
            Step(
                id=2,
                step_description="Find correlations between serotype and source of isolation in the dataset.",
                datasets=[],
                rationale="Understanding correlations between serotype and source of isolation can provide insights into the epidemiology of the pathogens in the dataset.",
                task_type=["summary_statistics", "visualization"],
                dependencies=[1]
            ),
        ],
        wants="Clarification on which serotypes or sources are of most interest.",
        misc="This is a sample plan for testing purposes."
    )
    
    test_executor(sample_plan)
