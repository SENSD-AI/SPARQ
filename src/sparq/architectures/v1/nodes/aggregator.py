from typing import List

import tiktoken

from sparq.schemas.state import State
from sparq.schemas.output_schemas import StepResult
from sparq.settings import LLMSetting
from sparq.utils.get_llm import get_llm

from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.messages import BaseMessage

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


def aggregator_node(state: State, llm_config: LLMSetting, prompt: str):
    results: List[StepResult] = state.results

    if not results:
        return {'answer': '[AGGREGATOR]: There are no results to return'}

    llm = get_llm(model=llm_config.model_name, provider=llm_config.provider)

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

    print("[AGGREGATOR]: Got results. Writing final report...")
    response: BaseMessage = llm.invoke(system_prompt_str)

    return {'answer': response.content}
