import json
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import AIMessage

from sparq.architectures.v1.reporting.bundle import organize_report
from sparq.architectures.v1.reporting.markdown import ReportMetadata
from sparq.architectures.v1.settings import V1LLMSettings
from sparq.logging_config import logger
from sparq.schemas.state import State
from sparq.settings import LLMSetting
from sparq.utils.helpers import pydantic_encoder


def _model_label(setting: LLMSetting | None) -> str:
    """Identify a configured model without including credentials or other settings."""
    if setting is None:
        return "Not recorded"
    return f"{setting.model_name} ({setting.provider})"


def saver_node(state: State, save_dir: Path, llm_config: V1LLMSettings | None = None):
    with logger.contextualize(node="saver"):
        # save only query and final answer
        final_message = state.messages[-1] if state.messages else None
        state_concise = {
            'query': state.query,
            'answer': final_message.text if isinstance(final_message, AIMessage) else None,
        }
        save_path = save_dir / 'final_answer.json'
        with open(save_path, 'w') as file:
            json.dump(state_concise, file, indent=4, default=pydantic_encoder)
        if isinstance(final_message, AIMessage) and final_message.text.strip():
            direct_answer = state.route is False
            author_role = "router" if direct_answer else "aggregator"
            author_setting = getattr(llm_config, author_role) if llm_config else None
            metadata = ReportMetadata(
                question=state.query,
                question_asked_at=state.question_asked_at,
                generated_at=datetime.now(UTC),
                author_model=_model_label(author_setting),
                author_role=author_role,
                planner_model=(
                    "Not used"
                    if direct_answer
                    else _model_label(llm_config.planner if llm_config else None)
                ),
                executor_model=(
                    "Not used"
                    if direct_answer
                    else _model_label(llm_config.executor if llm_config else None)
                ),
            )
            report_dir = organize_report(final_message.text, save_dir, metadata)
            logger.info('Saved final report to {}', report_dir)
