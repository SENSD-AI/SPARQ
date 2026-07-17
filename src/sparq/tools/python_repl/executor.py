from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
import json
import multiprocessing as mp
import os
import pickle
import tempfile
import traceback
import types

import ast

from sparq.tools.python_repl.ast_utils import rewrite_last_expr, compile_for_repl
from sparq.tools.python_repl.namespace import load_ns, clean_namespace, get_modules_in_namespace
from sparq.tools.python_repl.package_manager import PackageUtils as putils
from sparq.tools.python_repl.schemas import OutputSchema, ExceptionInfo


def pickle_vars(namespace: dict) -> dict[str, object]:
    """
    Returns a dictionary of all picklable objects in the namespace (new and modified).
    If an object cannot be pickled, it is added to the "__unpicklable__" key with its type name.
    """
    new_objs = {}
    unpickleable = {}

    for key, value in namespace.items():
        # Skip modules since they can't be pickled and will be re-imported in the subprocess
        if isinstance(value, types.ModuleType):
            continue

        try:
            pickle.dumps(value)
            new_objs[key] = value
        except (pickle.PicklingError, TypeError, AttributeError):
            unpickleable[key] = type(value).__name__

    if unpickleable:
        new_objs["__unpicklable__"] = unpickleable
    return new_objs


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


def execute_code(code: str, ns_path: str | None = None, timeout: int = 2*60) -> OutputSchema:
    """
    Execute Python code with optional namespace persistence and timeout.

    Args:
        code: Python code to execute
        ns_path: Path to a persistent namespace pickle file. If None, a temporary namespace is used and deleted after execution.
        timeout: Maximum execution time in seconds

    Returns:
        OutputSchema containing execution results, including output, errors, and a
        JSON-safe namespace summary (complex objects shown as type strings).
    """
    if ns_path is not None:
        # namespace should be persisted
        ns_is_temp = False
    else:
        # namespace should not be persisted
        ns_fd, ns_path = tempfile.mkstemp(suffix="_ns.pkl")
        with os.fdopen(ns_fd, "wb") as f:
            pickle.dump({}, f)
        ns_is_temp = True


    result_fd, result_path = tempfile.mkstemp(suffix="_result.json")
    os.close(result_fd)

    result = _execute_code_in_new_process(code, timeout=timeout, ns_path=ns_path, result_path=result_path)

    # On import error, install the package if whitelisted and retry execution
    # TODO: Make max_retries a global if possible
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

    # Unlink temporary namespace (if created) and result files
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
    # Spawn a fresh child process per execution for full isolation and timeout control.
    # Syntax errors are caught naturally inside _target via ast.parse.
    ctx = mp.get_context("spawn")
    process = ctx.Process(target=_target, args=(code, ns_path, result_path))
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


def _target(code: str, ns_path: str, result_path: str) -> None:
    """
    Target function run inside the spawned subprocess.

    Execution flow:
      1. Load the persisted namespace from disk.
      2. Re-import any modules tracked from prior executions (modules can't be
         pickled, so they're stored by name and re-imported here).
      3. Parse the code, rewrite the last bare expression for value capture,
         and compile with linecache registration for human-readable tracebacks.
      4. exec() the compiled code object.
      5. On success: merge updated variables back into the namespace pickle file.
      6. Write a JSON result to result_path.
    """
    # Force a non-interactive backend so plt.show() can't pop up a GUI window
    # from this headless subprocess; must be set before matplotlib is imported.
    os.environ.setdefault("MPLBACKEND", "Agg")

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
            namespace[var_name] = importlib.import_module(module_name)
        except ImportError:
            pass

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Parse → rewrite last expr for value capture → compile with linecache.
            # ast.parse raises SyntaxError here if the code is invalid; it is caught
            # by the outer except block and written to result_path like any other error.
            tree = ast.parse(code)
            tree, has_result = rewrite_last_expr(tree)
            code_obj = compile_for_repl(code, tree)
            exec(code_obj, namespace)

            # Retrieve the last-expression value stored by the AST rewrite.
            # A sentinel (not None) distinguishes three cases:
            #   a) No bare expression in code          → repl_result is _sentinel
            #   b) Expression evaluated to None        → repl_result is None
            #   c) Expression evaluated to some value  → repl_result is that value
            # Without a sentinel, cases (a) and (b) look identical.
            _sentinel = object()
            repl_result = namespace.pop("__repl_result__", _sentinel) if has_result else _sentinel

            stdout_text = stdout_buffer.getvalue().strip()
            if has_result and repl_result is not _sentinel:
                # Expression was present and evaluated successfully
                output = (stdout_text if stdout_text else "None") if repl_result is None else str(repl_result)
            else:
                # No bare expression — output is whatever was printed to stdout
                output = stdout_text if stdout_text else ""

        stderr_text = stderr_buffer.getvalue().strip()
        if stderr_text:
            output = f"{output}\n[stderr]: {stderr_text}" if output else f"[stderr]: {stderr_text}"

        # Build picklable objects for all variables in the namespace (new and modified)
        clean_namespace(namespace)
        new_objs = pickle_vars(namespace)
        mods = get_modules_in_namespace(namespace)
        if mods:
            new_objs["__modules__"] = mods

        # Merge new vars into the namespace file
        existing_ns = load_ns(ns_path)
        existing_ns.update(new_objs)
        with open(ns_path, "wb") as f:
            pickle.dump(existing_ns, f)

        result = OutputSchema(
            output=output,
            error=None,
            success=True,
            namespace=_namespace_summary(new_objs),
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
