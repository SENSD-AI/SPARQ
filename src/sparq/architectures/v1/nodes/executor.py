from pathlib import Path
import asyncio
from typing import List, Tuple

from sparq.schemas.state import State
from sparq.schemas.output_schemas import Plan, Step, StepResult
from sparq.settings import LLMSetting
from sparq.utils import helpers
from sparq.utils.get_llm import get_llm
from sparq.logging_config import logger
from sparq.tools.python_repl.python_repl_tool import make_python_repl_tool
from sparq.tools.python_repl.namespace import get_ns_path, load_ns, save_ns
from sparq.tools.filesystemtools import filesystemtools
from sparq.tools.data_discovery_tools import make_load_dataset_tool, get_sheet_names, find_csv_excel_files, get_cached_dataset_path

from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

MAX_NAMESPACE_VARS_WARNING = 30
MAX_RETRY = 3


def _build_context(dependency_results: List[StepResult], ns_context: str) -> str:
    """
    Build a context string for a worker's current step based on its dependencies' results
    and the variables already available in its namespace.

    Args:
        dependency_results: Results of the steps this step depends on.
        ns_context: Summary of variables already in the worker's namespace (from merge_namespaces_of_previous_deps).
    """
    parts = []

    if dependency_results:
        lines = ["Results from steps this step depends on:"]
        for dep in dependency_results:
            lines.append(f"\nStep {dep.id}: {dep.step}")
            lines.append(f" Step Successful: {dep.success}")
            lines.append(f"  Results: {dep.execution_results}")
            if dep.misc:
                lines.append(f"  Notes: {dep.misc}")
        parts.append("\n".join(lines))

    if ns_context:
        parts.append(ns_context)

    return "\n\n".join(parts)

def merge_namespaces_of_previous_deps(step: Step, run_id: str) -> Tuple[int, str]:
    """Merge the namespaces of a step's completed dependencies into its own namespace.

    Returns (number of variables available, summary of those variables as "name: type" lines)
    for use in the worker's prompt, and so callers can flag context bloat if there are too many.
    """
    dependencies: List[int] = step.dependencies
    merged_namespace: dict = {}
    merged_modules: dict = {}

    for dep in dependencies:
        step_run_id: str = f"{run_id}_step_{dep}"
        namespace = load_ns(get_ns_path(step_run_id))
        merged_modules.update(namespace.pop("__modules__", {}))
        merged_namespace.update(namespace)

    if merged_modules:
        merged_namespace["__modules__"] = merged_modules

    if merged_namespace:
        save_ns(get_ns_path(f"{run_id}_step_{step.id}"), merged_namespace)

    ns_vars = {k: type(v).__name__ for k, v in merged_namespace.items() if not k.startswith("__")}
    num_vars = len(ns_vars)
    if num_vars == 0:
        return 0, ""

    lines = [f"Variables already available in your namespace ({num_vars} from completed dependency steps):"]
    for var, typename in ns_vars.items():
        lines.append(f"  {var}: {typename}")
    return num_vars, "\n".join(lines)

def get_results_of_dependent_steps(step: Step, state: State) -> List[StepResult]:
    return [step_result for step_result in state.results if step_result.id in step.dependencies]


async def executor_node(state: State, config: RunnableConfig, llm_config: LLMSetting, worker_prompt: str, output_dir: Path):
    """
    Execute the plan.

    Args:
        state: LangGraph state containing the plan and data context.
        config: LangGraph RunnableConfig; must contain config["configurable"]["run_id"] to scope the REPL namespace.
        llm_config: LLM model and provider settings for worker agents.
        worker_prompt: System prompt for a worker agent.
        output_dir: Directory where executor outputs (plots, files) are written.
    """
    with logger.contextualize(node="executor"):
        logger.info("Executing plan to answer your query...")
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = config.get('configurable', {}).get('run_id', 'default')

        plan = state.plan
        completed = set() # Why a set instead of a list? -> Downstream checking for completed steps is faster on a set than a list
        all_results: List[StepResult] = []
        attempt_counts: dict[int, int] = {} # Tracking the attempt counts for each step

        while len(completed) < len(plan.steps):
            # Stage 1: Add steps that are ready to be executed to a list
            ready_steps: List[Step] = []

            for step in plan.steps:
                if step.id in completed:
                    continue

                # Check if step has deps. If yes, then check if they have been completed. If no, then append step to ready_steps
                if step.dependencies:
                    # Check if dependencies are completed
                    if all(dep in completed for dep in step.dependencies):
                        ready_steps.append(step)
                else:
                    ready_steps.append(step)

            if not ready_steps:
                raise ValueError("Deadlock detected in executor node: Unresolved dependencies remain.")

            logger.info(f"Spawning {len(ready_steps)} parallel worker agents...")

            # Step 2: Execute the steps that are ready
            tasks = []
            for step in ready_steps:
                dependency_results: List[StepResult] = get_results_of_dependent_steps(step, state)
                tasks.append(
                    execute_single_step_worker(
                        step=step,
                        run_id=run_id,
                        llm_config=llm_config,
                        prompt=worker_prompt,
                        dependency_results=dependency_results,
                        state=state,
                        output_dir=output_dir
                    )
                )

            # Step 3: Track steps that were completed successfully and return them
            batch_results: List[StepResult] = await asyncio.gather(*tasks)

            for result in batch_results:
                if result.success:
                    completed.add(result.id)
                    all_results.append(result)
                else:
                    attempt_counts[result.id] = attempt_counts.get(result.id, 0) + 1
                    if attempt_counts[result.id] >= MAX_RETRY:
                        logger.warning(f"Step {result.id} failed after {attempt_counts[result.id]} retries - accepting as terminal failure.")
                        completed.add(result.id)
                        all_results.append(result)
                    else:
                        logger.warning(f"Step {result.id} failed (attempt {attempt_counts[result.id]}/{MAX_RETRY}), retrying...")

            # Keep `state` in sync so the next batch's get_results_of_dependent_steps sees this round's results.
            state = state.model_copy(update={"completed_plan_steps": list(completed), "results": all_results})

        return {"completed_plan_steps": list(completed), "results": all_results}


async def execute_single_step_worker(step: Step, run_id: str, llm_config: LLMSetting, prompt: str, dependency_results: List[StepResult], state: State, output_dir: Path) -> StepResult:
    """An isolated worker instance spawned for an individual plan step"""
    logger.info(f"[Worker Agent] starting step {step.id}: {step.step_description}")

    # Get namespace for worker, seeded with the merged namespaces of its dependencies
    step_ns_id = f"{run_id}_step_{step.id}"
    worker_ns: str = get_ns_path(step_ns_id)
    num_vars, ns_context = merge_namespaces_of_previous_deps(step, run_id)
    if num_vars > MAX_NAMESPACE_VARS_WARNING:
        logger.warning(f"[Worker Agent] step {step.id}: {num_vars} variables inherited from dependencies, exceeds {MAX_NAMESPACE_VARS_WARNING} — risk of context bloat.")

    # Create Deep Agent
    llm_object = get_llm(llm_config.model_name, llm_config.provider)
    data_context = state.data_context
    _tools = [
        make_load_dataset_tool(worker_ns),
        get_sheet_names,
        make_python_repl_tool(worker_ns),
        find_csv_excel_files,
        get_cached_dataset_path,
    ] + (filesystemtools(working_dir=str(output_dir), selected_tools='all'))

    system_prompt_template: BasePromptTemplate = PromptTemplate.from_template(prompt).partial(
        data_context=str(data_context),
        output_dir=str(output_dir)
    )
    system_prompt_str: str = system_prompt_template.invoke(input={}).to_string()
    system_prompt: SystemMessage = SystemMessage(content=system_prompt_str)

    agent = create_agent(
        model=llm_object,
        name=f"Agent {step.id}",
        tools=_tools,
        middleware=[TodoListMiddleware()],
        system_prompt=system_prompt,
        response_format=StepResult,
    )

    context = _build_context(dependency_results, ns_context)
    user_content = f"{context}\n\nYour current task:\n{step.step_description}" if context else step.step_description
    agent_input = {"messages": [{"role": "user", "content": user_content}]}

    try:
        response = await agent.ainvoke(agent_input, config={"recursion_limit": llm_config.recursion_limit})
        result: StepResult = response["structured_response"]
    except Exception as e:
        result = StepResult(id=step.id, step=step.step_description, success=False, execution_results="", misc=f"Step failed: {e}")

    result.id = step.id
    result.step = step.step_description

    logger.info(f"[Worker Agent] completed step {step.id}: {step.step_description}")
    return result


async def test_executor(plan: Plan):
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
    state = State(query="", route=None, answer=None, plan=plan, data_context=data_context)
    response = await executor_node(state=state, config={"configurable": {"run_id": "test"}}, llm_config=system_settings.llm_config.executor, worker_prompt=prompt, output_dir=run_dir)

    for result in response['results']:
        print(result.id)
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
                dependencies=[]
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
    
    asyncio.run(test_executor(sample_plan))
