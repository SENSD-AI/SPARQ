from pathlib import Path

import tiktoken
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import BasePromptTemplate, PromptTemplate

from sparq.logging_config import logger
from sparq.schemas.output_schemas import StepResult
from sparq.schemas.state import State
from sparq.settings import LLMSetting
from sparq.tools.artifact_tools import ArtifactTools
from sparq.tools.filesystemtools import filesystemtools
from sparq.tools.image_processing.image_analysis_tool import make_image_analysis_tool
from sparq.utils.get_llm import get_llm

# Used when llm_config.max_tokens is not set. LLMSetting has no per-model context
# window field yet (see docs/improvements.md #13), so this is a conservative guess.
DEFAULT_CONTEXT_WINDOW = 128_000

# cl100k_base is a reasonable proxy across providers; it's a local BPE tokenizer
# (no API call), unlike some providers' get_num_tokens() (e.g. google_genai).
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def format_results(results: list[StepResult]) -> str:
    """Format step results as labeled text for the aggregator prompt.

    Each result includes all serialized fields, with the ID on the same line
    as its label and other values on the following line. Results retain their
    input order and are separated by a horizontal line of hyphens.

    Args:
        results: Execution results to include in the prompt.

    Returns:
        The formatted results, or an empty string if no results are supplied.
    """
    parts = []
    for result in results:
        part_str = ""
        for key, value in result.model_dump().items():
            part_str += f"{key}:{value}\n" if key == "id" else f"{key}:\n{value}\n"
        parts.append(part_str)

    return "\n-----------------------------------------------\n".join(parts)


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))

# TODO: implement this
def truncate_results(results: list[StepResult], max_tokens: int) -> list[StepResult]:
    """Iteratively truncate the least important step content first (misc ->
    files_generated -> step), then shrink execution_results text, until the
    formatted results fit within max_tokens. See docs/improvements.md #13.
    """
    pass


async def aggregator_node(
    state: State,
    llm_config: LLMSetting,
    prompt: str,
    working_dir: Path,
    vision_model: BaseChatModel,
):
    with logger.contextualize(node="aggregator"):
        results: list[StepResult] = state.results

        if not results:
            return {'messages': [AIMessage(content='[AGGREGATOR]: There are no results to return')]}

        max_tokens = llm_config.max_tokens or DEFAULT_CONTEXT_WINDOW
        execution_results = format_results(results)
        if count_tokens(execution_results) > max_tokens:
            results = truncate_results(results, max_tokens)
            execution_results = format_results(results)

        system_prompt_template: BasePromptTemplate = PromptTemplate.from_template(prompt).partial(
            user_query=state.query,
            # plan=str(state['plan']),
            execution_results=execution_results
        )

        system_prompt_str: str = system_prompt_template.invoke(input={}).to_string()

        llm = get_llm(model=llm_config.model_name, provider=llm_config.provider)
        artifact_tools = ArtifactTools(working_dir, index_name='executor/index.json')
        agent = create_agent(
            model=llm,
            name="aggregator",
            tools=[
                tool(artifact_tools.search_artifact),
                tool(artifact_tools.list_artifacts),
                make_image_analysis_tool(vision_model, image_root=working_dir),
            ] + filesystemtools(working_dir=str(working_dir), selected_tools=['read_file', 'list_directory', 'file_search']),
            system_prompt=system_prompt_str
        )

        logger.info("Got results. Writing final report...")
        response = await agent.ainvoke({"messages": [{"role": "user", "content": "Write the final report now."}]})
        final_message = response["messages"][-1]

    return {'messages': [final_message]}
