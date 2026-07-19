from sparq.schemas.output_schemas import Plan
from sparq.schemas.state import State
from sparq.logging_config import logger
from sparq.utils.helpers import pydantic_encoder

import json
from pathlib import Path

def saver_node(state: State, save_dir: Path):
    with logger.contextualize(node="saver"):
        # save only query and final answer
        keys = ['query', 'answer']
        state_concise = {key: getattr(state, key) for key in keys if hasattr(state, key)}
        save_path = save_dir / 'final_answer.json'
        with open(save_path, 'w') as file:
            json.dump(state_concise, file, indent=4, default=pydantic_encoder)
