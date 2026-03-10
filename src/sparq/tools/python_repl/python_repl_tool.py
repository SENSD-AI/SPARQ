from langchain.tools import tool

from sparq.tools.python_repl.schemas import PythonREPLInput
from sparq.tools.python_repl.executor import execute_code
from sparq.tools.python_repl.schemas import OutputSchema


@tool(args_schema=PythonREPLInput, response_format='content_and_artifact')
def python_repl_tool(code: str = "", persist_namespace: bool = False) -> tuple[str, OutputSchema]:
    """
    Executes the given Python code in a REPL environment.
    Supports variable persistence across executions and automatic installation of 
    white-listed packages, if missing.

    Args:
        code: The Python code to execute. Default is an empty string.
        persist_namespace: Whether to persist the namespace across executions. Default is False.

    Returns:
        str: The formatted message is shown to the LLM.
    """
    execution_result = execute_code(code, persist_namespace=persist_namespace)
    
    # Create clean message for LLM
    if execution_result.success:
        response = f"✓ Code executed successfully.\nOutput:\n{execution_result.output}"
    else:
        response = f"✗ Execution failed.\nError ({execution_result.error.type}): {execution_result.error.message}\n\nTraceback:\n{execution_result.error.traceback}\nExtra Context:\n{execution_result.error.extra_context}"
    
    return response, execution_result