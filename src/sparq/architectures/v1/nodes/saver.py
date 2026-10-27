import json
from pathlib import Path

from langchain_core.messages import AIMessage

from sparq.logging_config import logger
from sparq.schemas.output_schemas import Plan
from sparq.schemas.state import State
from sparq.utils.helpers import pydantic_encoder


def saver_node(state: State, save_dir: Path):
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
