from sparq.schemas.state import State
from sparq.schemas.output_schemas import Plan, ExecutorOutput
from sparq.settings import Settings
from sparq.utils import helpers
from sparq.tools.python_repl.python_repl_tool import python_repl_tool
from sparq.tools.filesystemtools import filesystemtools
from sparq.tools.data_discovery_tools import load_dataset, get_sheet_names, find_csv_excel_files, get_cached_dataset_path

from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

def executor_node(state: State, **kwargs):
    """
    Execute the plan
    """
    print("Executing plan to answer your query.")
    
    plan: Plan = state['plan']
    llm = kwargs['llm']
    prompt = kwargs['prompt']
    output_dir = kwargs['output_dir']
    
    data_manifest = state['data_manifest']
    df_summaries = state['df_summaries']
    
    _tools = [
        load_dataset,
        get_sheet_names,
        python_repl_tool,
        find_csv_excel_files,
        get_cached_dataset_path,
    ] + (filesystemtools(working_dir=str(output_dir), selected_tools='all'))
    
    system_prompt_template: BasePromptTemplate = PromptTemplate.from_template(prompt).partial(
        data_manifest=str(data_manifest),
        df_summaries=str(df_summaries),
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
    
    # FIXME: Grab the code from tool call args and results from tool call results (OutputSchema)
    def process_step(results: dict, step_description, step_index, prior_messages: list):
        agent_input = {"messages": prior_messages + [{"role": "user", "content": step_description}]}
        response = agent.invoke(agent_input)
        structured_response = response["structured_response"]
        
        # store response in results_dict
        outer_dict_key = f"Step {step_index}: {step_description}"
        results[outer_dict_key] = {} # make each key a dict
        inner_dict = results[outer_dict_key] # create reference to inner dict
        
        inner_dict['code'] = structured_response.code if structured_response.code is not None else ""
        inner_dict['previously_done'] = structured_response.previously_done if structured_response.previously_done is not None else ""
        inner_dict['execution_results'] = structured_response.execution_results
        inner_dict['files_generated'] = structured_response.files_generated
        inner_dict['assumptions'] = structured_response.assumptions
        inner_dict['wants'] = structured_response.wants
        inner_dict['misc'] = structured_response.misc    
        
        return results, response["messages"]
    
    results = {}
    prior_messages = []
    for i, step in enumerate(plan.steps):
        step_description = step.step_description
        results, prior_messages = process_step(results, step_description, i+1, prior_messages)
        state['executor_results'] = results

    return state

def test_executor(plan: Plan):
    print(f"Testing executor node with sample plan: \n {plan.pretty_print()}")
    
    settings = Settings()
    llm = helpers.get_llm(model='o3-mini')
    prompt = helpers.load_text(settings.EXECUTOR_PROMPT_PATH)
    output_dir = settings.EXECUTOR_OUTPUT_DIR
    
    manifest_path = settings.DATA_MANIFEST_PATH
    manifest = helpers.load_data_manifest(manifest_path)
    df_summaries = helpers.get_df_summaries_from_manifest(manifest)

    state = {'plan': plan, 'data_manifest': manifest,'df_summaries': df_summaries}
    response = executor_node(state=state, prompt=prompt, output_dir=output_dir, llm=llm)
    
    for step, result in response['executor_results'].items():
        print(step)
        print(result)

if __name__ == "__main__":
    from sparq.schemas.output_schemas import Step

    sample_plan = Plan(
        steps=[
            Step(
                step_description="Determine which dataset(s) to use",
                datasets=[],
                rationale="Not all datasets are relevant.",
                task_type=["data_mining"]
            ),
            Step(
                step_description="Summarize the distribution of serotypes in the dataset.",
                datasets=[],
                rationale="Understanding serotype distribution is crucial for epidemiological insights.",
                task_type=["summary_statistics", "visualization"]
            ),
        ],
        wants="Clarification on which serotypes or sources are of most interest.",
        misc="This is a sample plan for testing purposes."
    )
    
    test_executor(sample_plan)