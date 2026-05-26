from pydantic import BaseModel
from sparq.schemas.output_schemas import Plan
from sparq.schemas.state import State

import json
from pathlib import Path

def pydantic_encoder(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode='json')
    
    raise TypeError(f"Object of type {__obj.__class__.__name__} is not JSON serializable")

def saver_node(state: State, save_dir: Path):
    # save entire trace
    save_path = save_dir / 'trace.json'
    with open(save_path, 'w') as file:
        json.dump(state, file, indent=4, default=pydantic_encoder)

    # save only query and final answer
    keys = ['query', 'answer']
    state_concise = {key:state[key] for key in keys if key in state}
    save_path = save_dir / 'final_answer.json'
    with open(save_path, 'w') as file:
        json.dump(state_concise, file, indent=4, default=pydantic_encoder)
