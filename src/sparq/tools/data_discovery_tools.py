from langchain_core.tools import tool, InjectedToolCallId
from langchain_community.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langgraph.types import Command
from langchain_core.messages import ToolMessage

import pickle
from sparq.tools.python_repl.namespace import get_persistent_ns_path, load_ns


from pathlib import Path
from typing import Annotated

@tool
def load_dataset(file_path, sheet_name=None, var_name: str = "df"):
    """
    Loads a dataset from either a CSV or an Excel sheet.
    Args:
        file_path (str): Path to the dataset file.
        sheet_name (str, optional): Name of the Excel sheet to load. Defaults to None.
        var_name (str, optional): The variable name to assign the loaded dataset to in the persistent namespace. Defaults to "df".
    Returns:
    """
    import pandas as pd

    if file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return f"PythonError: {e}"
    elif file_path.endswith('.xlsx') and sheet_name:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            return f"PythonError: {e}"
    else:
        raise ValueError("Unsupported file format or missing sheet name for Excel file.")

    ns_path = get_persistent_ns_path()
    ns = load_ns(ns_path)
    ns[var_name] = df
    with open(ns_path, "wb") as f:
        pickle.dump(ns, f)

    return f"Loaded dataset into variable `{var_name}`.\n\nPreview:\n{df.head().to_markdown()}"

@tool
def get_sheet_names(file_path):
    """
    Returns the sheet names of an Excel file. (Only works if argument is an excel file)
    Args:
        file_path (str): The path to the Excel file.
    Returns:
        list: A list of sheet names.
    """
    import pandas as pd
    
    sheet_names = "Sheet names:\n"
    if not file_path.endswith('.xlsx'):
        return ("Provided file is not an excel file.")
    
    try:
        excel_file = pd.ExcelFile(file_path)
    except Exception as e:
        return f"PythonError: {e}"
    
    for sheet_name in excel_file.sheet_names:
        sheet_names += f"- {sheet_name}\n"
        
    return sheet_names


@tool
def get_cached_dataset_path(repo_id: str):
    """ Get the path to a cached dataset from Hugging Face Hub.
    Args:
        repo_id (str): The repository ID of the dataset on Hugging Face Hub.
    Returns:
        Path: The path to the cached dataset. (If dataset doesn't exist, it will be attempted to be downloaded.)
    """
    from huggingface_hub import snapshot_download
    import os
    from pathlib import Path
    
    # Load HF Token
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is not set. Please set it before running the script.")
    
    # get path to cached dataset
    try:
        path = snapshot_download(repo_id=repo_id, repo_type="dataset", token=HF_TOKEN, local_files_only=True)
    except Exception as e:
        raise FileNotFoundError(f"Dataset with repo_id {repo_id} not found in cache. Please download it first.") from e

    return Path(path)

@tool
def find_csv_excel_files(root_dir: Path | str) -> list[Path]:
    """
    Recursively find all CSV and Excel files in a directory.

    This tool is especially useful for discovering datasets in a directory when the exact file names are not known. It searches for files with .csv, .xls, and .xlsx extensions.
    For example, data inside huggingface cache directories often have unpredictable names, so this tool can help locate the relevant dataset files.
    
    Args:
        root_dir (Path): Root directory to search.
    
    Returns:
        List[Path]: List of file paths with .csv, .xls, or .xlsx extensions.
    """
    from pathlib import Path
    
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)
    
    if not root_dir.is_dir():
        raise NotADirectoryError(f"{root_dir} is not a valid directory")

    exts = {'.csv', '.xls', '.xlsx'}
    return [f for f in root_dir.rglob("*") if f.suffix.lower() in exts]
