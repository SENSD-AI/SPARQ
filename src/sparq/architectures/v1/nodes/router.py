from sparq.schemas.state import State
from sparq.schemas.output_schemas import Router
from sparq.settings import LLMSetting
from sparq.utils.get_llm import get_llm

from typing import cast
from langchain_core.messages import SystemMessage, HumanMessage

def router_func(router_output):
    """
    Function to determine the route of the query
    """
    return router_output.route

def router_node(state: State, llm_config: LLMSetting, prompt: str):
    """
    Route the user query to the appropriate node based on the type of query
    """
    llm = get_llm(model=llm_config.model_name, provider=llm_config.provider)
    structured_llm = llm.with_structured_output(Router)
    response = cast(Router, structured_llm.invoke([SystemMessage(content=prompt), HumanMessage(content=state.query)]))

    return {
        'route': response.route,
        'answer': response.answer,
    }
