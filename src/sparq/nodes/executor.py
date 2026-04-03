from pathlib import Path

from sparq.schemas.state import State
from sparq.schemas.output_schemas import Plan, ExecutorOutput
from sparq.settings import LLMSetting
from sparq.utils import helpers
from sparq.utils.get_llm import get_llm
from sparq.tools.python_repl.python_repl_tool import python_repl_tool
from sparq.tools.python_repl.namespace import get_persistent_ns_path, load_ns
from sparq.tools.filesystemtools import filesystemtools
from sparq.tools.data_discovery_tools import load_dataset, get_sheet_names, find_csv_excel_files, get_cached_dataset_path

from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent


def _build_context(results: dict) -> str:
    """
    Build a context string for the current step based on previous results and the current namespace.

    What is added to the context?
    - Previously completed steps and their results
    - Notes or miscellaneous information from previous steps
    - Variables currently in the namespace and their types

    """
    lines = []

    if results:
        lines.append("Previously completed steps:")
        for step, data in results.items():
            lines.append(f"\n{step}")
            lines.append(f"  Results: {data['execution_results']}")
            if data['misc']:
                lines.append(f"  Notes: {data['misc']}")

    ns = load_ns(get_persistent_ns_path())
    ns_vars = {k: type(v).__name__ for k, v in ns.items() if not k.startswith("__")}
    if ns_vars:
        lines.append("\nVariables currently in namespace:")
        for var, typename in ns_vars.items():
            lines.append(f"  {var}: {typename}")

    return "\n".join(lines)


def executor_node(state: State, llm_config: LLMSetting, prompt: str, output_dir: Path):
    """
    Execute the plan
    """
    print("Executing plan to answer your query.")
    output_dir.mkdir(parents=True, exist_ok=True)

    plan: Plan = state['plan']
    llm = get_llm(model=llm_config.model_name, provider=llm_config.provider)

    data_context = state['data_context']

    _tools = [
        load_dataset,
        get_sheet_names,
        python_repl_tool,
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
        context = _build_context(results)
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
        state['executor_results'] = results

    return state

def test_executor(plan: Plan):
    print(f"Testing executor node with sample plan: \n {plan.pretty_print()}")

    from sparq.settings import (
        ENVSettings,
        AgenticSystemSettings,
        PathSettings,
        DATA_MANIFEST_PATH,
        DATA_SUMMARIES_SHORT_PATH,
    )
    from sparq.schemas.data_context import load_data_context

    # Load environment and system settings
    env_settings = ENVSettings()
    system_settings = AgenticSystemSettings()

    # Get system prompt
    prompt = helpers.load_text(system_settings.paths.prompts_dir / "executor_message.txt")
    updated_paths = PathSettings.model_validate(
        system_settings.paths.model_dump() | {"output_stem": "test_executor"}
    )
    run_dir = updated_paths.run_dir

    data_context = load_data_context(DATA_MANIFEST_PATH, DATA_SUMMARIES_SHORT_PATH)

    state = {'plan': plan, 'data_context': data_context}
    response = executor_node(state=state, llm_config=system_settings.llm_config.executor, prompt=prompt, output_dir=run_dir)
    
    for step, result in response['executor_results'].items():
        print(step)
        print(result)

if __name__ == "__main__":
    from sparq.schemas.output_schemas import Step

    sample_plan = Plan(
        steps=[
            Step(
                step_description="Get the pulsenet dataset and load it into a dataframe.",
                datasets=[],
                rationale="The pulsenet dataset contains information about various pathogens, including their serotypes and sources of isolation. Loading it into a dataframe will allow us to analyze the data and find correlations.",
                task_type=["data_mining"]
            ),
            Step(
                step_description="Find correlations between serotype and source of isolation in the dataset.",
                datasets=[],
                rationale="Understanding correlations between serotype and source of isolation can provide insights into the epidemiology of the pathogens in the dataset.",
                task_type=["summary_statistics", "visualization"]
            ),
        ],
        wants="Clarification on which serotypes or sources are of most interest.",
        misc="This is a sample plan for testing purposes."
    )
    
    test_executor(sample_plan)