from langchain.tools import tool

from sparq.tools.python_repl.schemas import PythonREPLInput
from sparq.tools.python_repl.executor import execute_code
from sparq.tools.python_repl.schemas import OutputSchema

def make_python_repl_tool(ns_path: str):
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

        Notes
        - Functions are not persisted across executions because they are not picklable. They are instead stored in the `__unpicklable__` key of the namespace.
        """
        execution_result = execute_code(code or "", ns_path=ns_path if persist_namespace else None)
        
        # Create clean message for LLM
        if execution_result.success:
            response = f"✓ Code executed successfully.\nOutput:\n{execution_result.output}"
        else:
            # Include the submitted code so the agent can see exactly what it wrote
            # alongside the traceback — without this it would have to reconstruct the
            # context from prior messages to understand what went wrong.
            response = (
                f"✗ Execution failed.\n\n"
                f"Code submitted:\n{code}\n\n"
                f"Error ({execution_result.error.type}): {execution_result.error.message}\n\n"
                f"Traceback:\n{execution_result.error.traceback}\n"
                f"Extra Context:\n{execution_result.error.extra_context}"
            )
        
        return response, execution_result

    return python_repl_tool