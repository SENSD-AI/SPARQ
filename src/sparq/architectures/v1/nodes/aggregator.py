from sparq.schemas.state import State
from sparq.settings import LLMSetting
from sparq.utils.get_llm import get_llm

from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.messages import BaseMessage

def aggregator_node(state: State, llm_config: LLMSetting, prompt: str):
    executor_results: dict = state['executor_results']
    llm = get_llm(model=llm_config.model_name, provider=llm_config.provider)

    system_prompt_template: BasePromptTemplate = PromptTemplate.from_template(prompt).partial(
        user_query=state['query'],
        # plan=str(state['plan']),
        execution_results=str(executor_results)
    )

    system_prompt_str: str = system_prompt_template.invoke(input={}).to_string()

    response: BaseMessage = llm.invoke(system_prompt_str)

    return {'answer': response.content}
    