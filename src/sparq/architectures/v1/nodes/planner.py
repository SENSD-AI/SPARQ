from sparq.schemas.state import State
from sparq.schemas.output_schemas import Plan
from sparq.schemas.data_context import DataContext, load_data_context
from sparq.settings import (
    BaseAgenticSettings,
    ENVSettings,
    DATA_MANIFEST_PATH,
    DATA_SUMMARIES_SHORT_PATH,
    LLMSetting,
)
from sparq.utils.get_llm import get_llm

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import BasePromptTemplate, PromptTemplate

def planner_node(state: State, llm_config: LLMSetting, sys_prompt: str):
    """
    Create a plan to answer the user query.

    Returns:
        dict: A dictionary containing the generated plan and data context.
    """
    print("Making a plan to answer your query")

    llm = get_llm(model=llm_config.model_name, provider=llm_config.provider)

    # load data context
    data_context = load_data_context(DATA_MANIFEST_PATH, DATA_SUMMARIES_SHORT_PATH)

    # create system prompt
    system_prompt_template: BasePromptTemplate = PromptTemplate.from_template(sys_prompt).partial(
        data_context=str(data_context),
    )
    _system_prompt: str = system_prompt_template.invoke(input={}).to_string()
    system_prompt: SystemMessage = SystemMessage(content=_system_prompt)

    structured_llm = llm.with_structured_output(Plan)
    plan = structured_llm.invoke([system_prompt, HumanMessage(content=state['query'])])

    print("Created plan")
    return {'plan': plan, 'data_context': data_context}

def test_planner():
    # from sparq.settings_old import Settings
    print("Running test code for planner.py")
    
    # s = Settings()
    # llm_config = s.LLM_CONFIG
    # llm = helpers.get_llm(model=llm_config['planner']['model'], provider=llm_config['planner']['provider'])

    _ = ENVSettings()  # Load environment variables
    llm_config = BaseAgenticSettings().llm_config
    system_prompt = "Create a plan to answer the user query"
    user_query = "What is the relation between time of day and traffic in Kuala Lumpur, Malaysia?"
    input = {"query": user_query}

    response = planner_node(state=input, llm_config=llm_config.planner, sys_prompt=system_prompt)
    response['plan'].pretty_print()
    
if __name__ == "__main__":
    test_planner()