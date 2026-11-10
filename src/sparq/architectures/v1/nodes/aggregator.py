from pathlib import Path

import tiktoken
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import BasePromptTemplate, PromptTemplate

from sparq.logging_config import logger
from sparq.schemas.planning import StepResult
from sparq.schemas.report import Report
from sparq.schemas.state import State
from sparq.settings import LLMSetting
from sparq.tools.artifact_tools import ArtifactTools
from sparq.tools.filesystemtools import filesystemtools
from sparq.tools.image_processing.image_analysis_tool import make_image_analysis_tool
from sparq.utils.get_llm import get_llm

# Input-results budget, independent of the model's max_tokens output limit.
# There is no per-model context-window field yet, so this is a conservative guess.
DEFAULT_CONTEXT_WINDOW = 128_000
MAX_REPORT_CHUNKS = 20

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
    raise NotImplementedError("Result truncation is not implemented.")


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

        max_input_tokens = DEFAULT_CONTEXT_WINDOW
        execution_results = format_results(results)
        if count_tokens(execution_results) > max_input_tokens:
            results = truncate_results(results, max_input_tokens)
            execution_results = format_results(results)

        system_prompt_template: BasePromptTemplate = PromptTemplate.from_template(prompt).partial(
            user_query=state.query,
            # plan=str(state['plan']),
            execution_results=execution_results
        )

        system_prompt_str: str = system_prompt_template.invoke(input={}).to_string()

        llm = get_llm(
            model=llm_config.model_name,
            provider=llm_config.provider,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            top_k=llm_config.top_k,
        )
        artifact_tools = ArtifactTools(working_dir, index_name='executor/index.json')
        agent = create_agent(
            model=llm,
            name="aggregator",
            tools=[
                tool(artifact_tools.search_artifact),
                tool(artifact_tools.list_artifacts),
                make_image_analysis_tool(vision_model, image_root=working_dir),
            ] + filesystemtools(working_dir=str(working_dir), selected_tools=['read_file', 'list_directory', 'file_search']),
            system_prompt=system_prompt_str,
            response_format=ToolStrategy(Report)
        )

        logger.info("Got results. Writing final report...")

        messages: list[AnyMessage] = [HumanMessage(content="Begin writing the report at your own pace.")]
        chunks: list[str] = []
        draft_path = working_dir / "report" / "draft.md"

        for chunk_number in range(1, MAX_REPORT_CHUNKS + 1):
            # Invoke the LM
            response = await agent.ainvoke({"messages": [*messages]})
            messages = response["messages"]

            # Check for and grab the last AI message
            last_ai_message = None
            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    last_ai_message = message
                    break
            if last_ai_message is None:
                raise ValueError("Report response contains no AI message.")

            metadata = last_ai_message.response_metadata
            stop_reason = metadata.get("stopReason") or metadata.get("stop_reason") or metadata.get("finish_reason")
            if stop_reason in {"max_tokens", "length", "max_output_tokens"}:
                raise ValueError("Report output was truncated; saved text remains in report/draft.md.")

            report = Report.model_validate(response["structured_response"])
            chunk = report.report.strip()
            if not chunk or chunk in chunks:
                raise ValueError("Report writing made no progress: empty or repeated text.")
            chunks.append(chunk)
            draft_path.parent.mkdir(parents=True, exist_ok=True)
            draft_path.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
            logger.info("Saved report chunk {} (done={})", chunk_number, report.done)
            if report.done:
                final_message = AIMessage(content="\n\n".join(chunks))
                break
            messages = [*messages, HumanMessage(content=(
                "Continue writing the report at your own pace from where you left off. "
                "Return only new text, and set done=true when the entire report is finished."
            ))]
        else:
            raise ValueError(f"Report did not finish within {MAX_REPORT_CHUNKS} responses; draft saved at {draft_path}.")

    return {'messages': [final_message]}
