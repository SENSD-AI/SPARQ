# utils.py
# Only stdlib imports at the top level — see docs/architecture/repl.md for why.
import json
from pathlib import Path
from typing import Dict, List, Optional


def pydantic_encoder(obj):
    from pydantic import BaseModel

    if isinstance(obj, BaseModel):
        return obj.model_dump(mode='json')

    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def write_json(save_dir: Path, file_name: str, object) -> None:
    """Write object to save_dir / file_name as JSON, overwriting any previous file."""
    from pydantic_core import to_jsonable_python

    content = json.dumps(to_jsonable_python(object, by_alias=False), indent=4)
    save_path = save_dir / file_name
    save_path.write_text(content, encoding="utf-8")


def load_text(file_path):
    """Loads text from a file."""
    with open(file_path, 'r') as f:
        text = f.read()

    return text


def get_user_query(args=None, config=None):
    if args is not None and args.test:
        if config is None:
            user_query = "What are the main factors contributing to salmonella rates in Missouri in a statistical sense?"
        else:
            user_query = config.get('test_query')
            if user_query is None:
                raise ValueError("Test query not found in config.")
            
        print("Using test query: ", user_query)
    else:
        user_query = input("Enter your query:\n")

    return user_query

def dump_dict_to_json(dict, save_path):
    """
    Dumps a dictionary to a JSON file.
    
    Args:
        dict (dict): The dictionary to dump.
        save_path (str): The path where the JSON file will be saved.
    """
    import json

    if isinstance(save_path, Path):
        save_path = str(save_path)

    try:
        with open(save_path, 'w') as f:
            json.dump(dict, f, indent=4)
        
        print(f"Dictionary successfully saved to {save_path}")
    except Exception as e:
        raise IOError(f"Could not write to file {save_path}:\n{e}")


def render_records_table(
    records: List[Dict],
    columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    no_wrap_columns: Optional[List[str]] = None,
) -> None:
    """Render a list of mapping records (list[dict]) to a table using rich and print it.

    This function prints the table to stdout and returns None. If a caller
    needs the rendered text instead, they should use a separate capture
    Console (not provided here).
    """
    from rich.console import Console
    from rich.table import Table
    console = Console()

    # Empty records -> print an empty table (with optional title)
    if not records:
        table = Table(title=title) if title else Table()
        console.print(table)
        return

    # Determine columns
    if columns is None:
        cols: List[str] = []
        for r in records:
            for k in r.keys():
                if k not in cols:
                    cols.append(k)
    else:
        cols = columns

    no_wrap_columns = no_wrap_columns or []
    table = Table(title=title, show_header=True, header_style="bold magenta")
    for c in cols:
        table.add_column(str(c), no_wrap=c in no_wrap_columns)

    for r in records:
        row = [str(r.get(c, "")) for c in cols]
        table.add_row(*row)

    console.print(table)
    return
