from pathlib import Path
from typing import List

import tiktoken

from sparq.schemas.state import State
from sparq.schemas.output_schemas import StepResult
from sparq.settings import LLMSetting
from sparq.utils.get_llm import get_llm
from sparq.tools.filesystemtools import filesystemtools

from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain.agents import create_agent

# Used when llm_config.max_tokens is not set. LLMSetting has no per-model context
# window field yet (see docs/improvements.md #13), so this is a conservative guess.
DEFAULT_CONTEXT_WINDOW = 128_000

# cl100k_base is a reasonable proxy across providers; it's a local BPE tokenizer
# (no API call), unlike some providers' get_num_tokens() (e.g. google_genai).
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def format_results(results: List[StepResult]) -> str:
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
def truncate_results(results: List[StepResult], max_tokens: int) -> List[StepResult]:
    """Iteratively truncate the least important step content first (misc ->
    files_generated -> step), then shrink execution_results text, until the
    formatted results fit within max_tokens. See docs/improvements.md #13.
    """
    pass


def aggregator_node(state: State, llm_config: LLMSetting, prompt: str, working_dir: Path):
    results: List[StepResult] = state.results

    if not results:
        return {'answer': '[AGGREGATOR]: There are no results to return'}

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
    agent = create_agent(
        model=llm,
        name="aggregator",
        tools=filesystemtools(working_dir=str(working_dir), selected_tools=['read_file', 'list_directory', 'file_search']),
        system_prompt=system_prompt_str
    )

    print("[AGGREGATOR]: Got results. Writing final report...")
    response = agent.invoke({"messages": [{"role": "user", "content": "Write the final report now."}]})
    answer: str = response["messages"][-1].content

    return {'answer': answer}
