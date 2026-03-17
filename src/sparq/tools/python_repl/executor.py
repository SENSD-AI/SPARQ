from contextlib import redirect_stderr, redirect_stdout
import io
import json
import multiprocessing as mp
import os
import pickle
import tempfile
import traceback
import types

from typing import Optional, List

from sparq.tools.python_repl.ast_utils import extract_last_expression
from sparq.tools.python_repl.namespace import get_persistent_ns_path, load_ns, clean_namespace, get_modules_in_namespace
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


def _namespace_summary(namespace: dict) -> dict:
    """JSON-safe summary of namespace variables for returning to the parent process.

    Complex objects (e.g. numpy arrays, DataFrames) are represented as type strings
    so the parent process never needs to import scientific packages to deserialize them.
    """
    summary = {}
    for key, value in namespace.items():
        try:
            json.dumps(value)
            summary[key] = value
        except (TypeError, ValueError):
            summary[key] = f"<{type(value).__module__}.{type(value).__name__}>"
    return summary


def execute_code(code: str, persist_namespace: bool = False, timeout: int = 2*60) -> OutputSchema:
    """
    Execute Python code with optional namespace persistence and timeout.

    Args:
        code: Python code to execute
        persist_namespace: Whether to persist variables across executions
        timeout: Maximum execution time in seconds

    Returns:
        OutputSchema containing execution results, including output, errors, and a
        JSON-safe namespace summary (complex objects shown as type strings).
    """
    if persist_namespace:
        ns_path = get_persistent_ns_path()
        ns_is_temp = False
    else:
        ns_fd, ns_path = tempfile.mkstemp(suffix="_ns.pkl")
        with os.fdopen(ns_fd, "wb") as f:
            pickle.dump({}, f)
        ns_is_temp = True

    result_fd, result_path = tempfile.mkstemp(suffix="_result.json")
    os.close(result_fd)

    result = _execute_code_in_new_process(code, timeout=timeout, ns_path=ns_path, result_path=result_path)

    # On import error, install the package if whitelisted and retry execution
    max_retries = 5
    retries = 0
    while result.error and result.error.type in ("ModuleNotFoundError", "ImportError") and retries < max_retries:
        missing_package = putils.extract_package_name_error(result.error.message)

        if not missing_package:
            break

        install_result = putils.install_package(missing_package)
        if not install_result["success"]:
            result.error.extra_context["package_install_failed"] = {
                "package": missing_package,
                "message": install_result["message"]
            }
            break

        result = _execute_code_in_new_process(code, timeout=timeout, ns_path=ns_path, result_path=result_path)
        retries += 1

    if ns_is_temp:
        try:
            os.unlink(ns_path)
        except FileNotFoundError:
            pass
    try:
        os.unlink(result_path)
    except FileNotFoundError:
        pass

    return result


def _execute_code_in_new_process(code: str, timeout: int = 10, ns_path: str = "", result_path: str = "") -> OutputSchema:
    """
    Execute Python code in a subprocess, reading/writing namespace via files.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        ns_path: Path to the namespace pickle file (read and updated by child)
        result_path: Path where the child writes a JSON result

    Returns:
        OutputSchema with output, error, success, and JSON-safe namespace summary.
        The parent process never deserializes numpy/pandas objects.
    """
    statements, expr, syntax_error = extract_last_expression(code)
    if syntax_error:
        return OutputSchema(
            output="",
            error=ExceptionInfo(type="SyntaxError", message=str(syntax_error), traceback="", extra_context={}),
            namespace={},
            success=False
        )

    ctx = mp.get_context("spawn")
    process = ctx.Process(target=_target, args=(statements, expr, ns_path, result_path))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return OutputSchema(
            output="",
            error=ExceptionInfo(
                type="TimeoutError",
                message="Code execution timed out.",
                traceback="",
                extra_context={"timeout_seconds": timeout}
            ),
            namespace={},
            success=False
        )

    try:
        with open(result_path, "r") as f:
            data = json.load(f)
        return OutputSchema(
            output=data["output"],
            error=ExceptionInfo(**data["error"]) if data["error"] else None,
            namespace=data["namespace"],
            success=data["success"]
        )
    except FileNotFoundError:
        return OutputSchema(
            output="",
            error=ExceptionInfo(type="ResultMissingError", message="Result file was not written.", traceback="", extra_context={}),
            namespace={},
            success=False
        )
    except Exception as e:
        return OutputSchema(
            output="",
            error=ExceptionInfo(type=type(e).__name__, message=str(e), traceback="", extra_context={}),
            namespace={},
            success=False
        )


def _target(statements: Optional[List[str]], expr: str, ns_path: str, result_path: str) -> None:
    """
    Target function for subprocess execution.

    - Reads namespace from ns_path
    - Executes code
    - On success: merges new variables back into ns_path
    - Writes a JSON result to result_path (never contains raw numpy/pandas objects)
    """
    result = OutputSchema(output="", error=None, success=False, namespace={})

    try:
        namespace = load_ns(ns_path)
    except Exception as e:
        result = OutputSchema(
            output="",
            error=ExceptionInfo(type=type(e).__name__, message=f"Failed to load namespace: {e}", traceback=traceback.format_exc(), extra_context={}),
            success=False,
            namespace={}
        )
        with open(result_path, "w") as f:
            json.dump(result.model_dump(), f)
        return

    # Re-import modules from previous executions
    for var_name, module_name in namespace.get("__modules__", {}).items():
        try:
            namespace[var_name] = __import__(module_name)
        except ImportError:
            pass

    original_keys = set(namespace.keys())
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            if statements:
                exec("\n".join(statements), namespace)

            output = ""
            if expr:
                eval_result = eval(expr, namespace)
                if eval_result is None:
                    stdout_text = stdout_buffer.getvalue().strip()
                    output = stdout_text if stdout_text else "None"
                else:
                    output = str(eval_result)
            else:
                stdout_text = stdout_buffer.getvalue().strip()
                output = stdout_text if stdout_text else ""

        stderr_text = stderr_buffer.getvalue().strip()
        if stderr_text:
            output = f"{output}\n[stderr]: {stderr_text}" if output else f"[stderr]: {stderr_text}"

        # Build picklable vars for new variables introduced in this execution
        clean_namespace(namespace)
        picklable_vars = pickle_vars(namespace, original_keys)
        mods = get_modules_in_namespace(namespace)
        if mods:
            picklable_vars["__modules__"] = mods

        # Merge new vars into the namespace file
        existing_ns = load_ns(ns_path)
        existing_ns.update(picklable_vars)
        with open(ns_path, "wb") as f:
            pickle.dump(existing_ns, f)

        result = OutputSchema(
            output=output,
            error=None,
            success=True,
            namespace=_namespace_summary(picklable_vars),
        )

    except Exception as e:
        clean_namespace(namespace)
        stdout_text = stdout_buffer.getvalue().strip()
        stderr_text = stderr_buffer.getvalue().strip()
        result = OutputSchema(
            output=stdout_text,
            error=ExceptionInfo(
                type=type(e).__name__,
                message=str(e),
                traceback=traceback.format_exc(),
                extra_context={"stderr": stderr_text} if stderr_text else {}
            ),
            success=False,
            namespace={},
        )

    finally:
        try:
            with open(result_path, "w") as f:
                json.dump(result.model_dump(), f)
        except Exception as e:
            print(f"Failed to write result: {e}")


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
