from contextlib import redirect_stderr, redirect_stdout
import io
import multiprocessing as mp
from queue import Empty
import traceback
import types
import pickle

from typing import Optional, List

from sparq.tools.python_repl.ast_utils import extract_last_expression
from sparq.tools.python_repl.namespace import get_persistent_namespace, clean_namespace, get_modules_in_namespace
from sparq.tools.python_repl.package_manager import PackageUtils as putils
from sparq.tools.python_repl.schemas import OutputSchema, ExceptionInfo

def pickle_vars(namespace: dict, original_keys: set) -> dict[str, object]:
    new_vars = {}
    unpickleable = {}

    for key in namespace:
        if key not in original_keys:
            value = namespace[key]

            # Skip modules since they can't be pickled and will be re-imported in the subprocess
            if isinstance(value, types.ModuleType):
                continue

            try:
                pickle.dumps(value)
                new_vars[key] = value
            except (pickle.PicklingError, TypeError, AttributeError):
                unpickleable[key] = type(value).__name__

    if unpickleable:
        new_vars["__unpicklable__"] = unpickleable
    return new_vars

def execute_code(code: str, persist_namespace: bool = False, timeout: int = 2*60) -> OutputSchema:
    """
    Execute Python code with optional namespace persistence and timeout.
    
    Args:
        code: Python code to execute
        persist_namespace: Whether to persist variables across executions
        timeout: Maximum execution time in seconds
        
    Returns:
        OutputSchema containing execution results, including output, errors, and namespace
    """
    if persist_namespace:
        namespace = get_persistent_namespace() # Use module-level namespace for persistence
    else:
        namespace = {} # Fresh namespace for non-persistent execution
    
    # Get modules from namespace to re-import in subprocess
    modules = namespace.get("__modules__", {})
    
    result = _execute_code_in_new_process(code, timeout=timeout, new_namespace=namespace, modules=modules, persist_namespace=persist_namespace)

    # On import error, install the package if whitelisted and retry execution
    # Loop to handle multiple missing packages (max 5 retries)
    max_retries = 5
    retries = 0
    while result.error and result.error.type in ("ModuleNotFoundError", "ImportError") and retries < max_retries:
        missing_package = putils.extract_package_name_error(result.error.message)

        # If no package found, break
        if not missing_package:
            break
            
        install_result = putils.install_package(missing_package)
        if not install_result["success"]:
            # Update error with installation failure info
            result.error.extra_context["package_install_failed"] = {
                "package": missing_package,
                "message": install_result["message"]
            }
            break
        
        # Retry execution after successful installation
        result = _execute_code_in_new_process(code, timeout=timeout, new_namespace=namespace, modules=modules, persist_namespace=persist_namespace)
        retries += 1
    
    return result


def _execute_code_in_new_process(code: str, timeout: int = 10, new_namespace: Optional[dict] = None, modules: Optional[dict] = None, persist_namespace: bool = False) -> OutputSchema:
    """
    Execute python code with optional namespace persistence and timeout.

    Args:
        code (str): The Python code to execute.
        timeout (int): Maximum time in seconds to allow for code execution.
        new_namespace (Optional[dict]): Namespace to use for code execution.
        modules (Optional[dict]): Modules to re-import in the execution namespace.
        persist_namespace (bool): Whether to persist the namespace after execution.

    Returns:
        OutputSchema: The result of the code execution, including output, error, and namespace.
    """

    # Extract the last expression from the code to evaluate it separately. Catch any syntax errors.
    statements, expr, syntax_error = extract_last_expression(code)
    if syntax_error:
        return OutputSchema(
            output="",
            error=ExceptionInfo(
                type="SyntaxError",
                message=str(syntax_error),
                traceback="",
                extra_context={}
            ),
            namespace=new_namespace or {},
            success=False
        )
    
    extra_time = 5
    ctx = mp.get_context("spawn") # New process context with no shared state
    queue = ctx.Queue()

    process = ctx.Process(
        target=_target, 
        args=(
            statements,
            expr,
            queue,
            new_namespace,
            modules or {},
            timeout + extra_time
            )
        )
    
    process.start()
    process.join(timeout)

    # If process is still alive after timeout, terminate it
    if process.is_alive():
        process.terminate()
        result = OutputSchema(
            output="",
            error=ExceptionInfo(
                type="TimeoutError",
                message="Code execution timed out.",
                traceback="",
                extra_context={"timeout_seconds": timeout}
            ),
            namespace=new_namespace or {},
            success=False
        )
    else:
        try:
            result = queue.get(timeout=5) # short timeout since process already finished
        except Empty:
            # Process finished but queue is empty
            result = OutputSchema(
                output="",
                error=ExceptionInfo(
                    type="QueueEmptyError",
                    message="Result queue was empty.",
                    traceback="",
                    extra_context={}
                ),
                namespace=new_namespace or {},
                success=False
            )
        except Exception as e:
            # Catch any other errors related to queue retrieval
            result = OutputSchema(
                output="",
                error=ExceptionInfo(
                    type=type(e).__name__,
                    message=str(e),
                    traceback="",
                    extra_context={}
                ),
                namespace=new_namespace or {},
                success=False
            )


    # Update the namespace if execution was successful and persistence is desired
    if result.success and persist_namespace:
        new_namespace.update(result.namespace)

    return result

def _target(statements: Optional[List[str]], expr: str, queue: mp.Queue, namespace: dict, modules: dict, timeout: int) -> None:
    """
    Target function for multiprocessing execution with stdout/stderr

    - Catches Any Exception (SyntaxError should be caught earlier)
    """
    # Re-import modules from previous executions
    for var_name, module_name in modules.items():
        try:
            namespace[var_name] = __import__(module_name)
        except ImportError:
            pass # Module not available. Skip

    # Track original keys
    original_keys = set(namespace.keys())

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # If there are statements, execute them with namespace 
            if statements:
                exec("\n".join(statements), namespace)

            output = ""
            if expr:
                eval_result = eval(expr, namespace)
                # If eval returns a value, convert it to string for output
                if eval_result is None:
                    stdout_text = stdout_buffer.getvalue().strip() # Capture any print statements when eval returns None
                    output = stdout_text if stdout_text else "None" # Eval result is None so return "None"
                else:
                    output = str(eval_result)
            else:
                # No expression to evaluate, capture stdout if available
                stdout_text = stdout_buffer.getvalue().strip()
                output = stdout_text if stdout_text else ""
        
        # Check stderr for any captured errors and append
        stderr_text = stderr_buffer.getvalue().strip()
        if stderr_text:
            output = f"{output}\n[stderr]: {stderr_text}" if output else f"[stderr]: {stderr_text}"

        # Clean the namespace by removing any built-in or special variables
        # Pickle variables (skip modules)
        # Get names of modules
        clean_namespace(namespace)
        picklable_vars = pickle_vars(namespace, original_keys)
        modules = get_modules_in_namespace(namespace)

        if modules:
            picklable_vars["__modules__"] = modules

        result = OutputSchema(
            output=output,
            error=None,
            namespace=picklable_vars, # Only include picklable variables in the namespace
            success=True
        )
    
    # Catches Any Exception (SyntaxError should be caught earlier)
    except Exception as e:
        # Clean the namespace by removing any built-in or special variables
        clean_namespace(namespace)

        # Collect any stdout and stderr output even in case of exception
        stdout_text = stdout_buffer.getvalue().strip()
        stderr_text = stderr_buffer.getvalue().strip()
        context_data = {"stderr": stderr_text} if stderr_text else {}

        result = OutputSchema(
            output=stdout_text,
            error=ExceptionInfo(
                type=type(e).__name__,
                message=str(e),
                traceback=traceback.format_exc(),
                extra_context=context_data
            ),
            namespace={}, # Return empty namespace on error since variables may be in inconsistent state
            success=False
        )

    finally:
        try:
            queue.put(result, timeout=timeout)
        except Exception as e:
            print(f"Failed to put result in queue: {e}")
            pass # Nothing we can do if putting result in queue fails

if __name__ == "__main__":    
    print("Testing tool python repl tool")

    code_snippet = """
x = 10
y = 20
x + y3
                    """
    output = execute_code(code_snippet, persist_namespace=True)
    print(output)

    # output = execute_code("x * 2", persist_namespace=True)
    # print(output)
