import ast
import linecache
import types
from typing import Tuple

from sparq.logging_config import get_logger

logger = get_logger(__name__)


def rewrite_last_expr(tree: ast.AST) -> Tuple[ast.AST, bool]:
    """
    If the last statement in the AST body is a bare expression, rewrite it to
    `__repl_result__ = <expr>` so exec() captures the value instead of discarding it.

    A bare expression is a statement that is purely a value — e.g. `df.head()` or
    `x + y` — with no assignment. exec() evaluates these but throws away the result.
    The rewrite turns the last such expression into an assignment to the sentinel
    variable `__repl_result__`, which the caller can retrieve from the namespace
    after exec() finishes.

    Returns:
        (tree, has_result) where has_result=True means the rewrite happened and
        `__repl_result__` will be present in the namespace after a successful exec.
    """
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        # Last statement is not a bare expression (e.g. it's an assignment,
        # a for-loop, a function def) — nothing to capture, return unchanged.
        return tree, False

    last = tree.body[-1]
    tree.body[-1] = ast.Assign(
        # ast.Store() signals this name is being written to (left-hand side of =)
        targets=[ast.Name(id="__repl_result__", ctx=ast.Store())],
        value=last.value,           # original expression becomes the right-hand side
        lineno=last.lineno,
        col_offset=last.col_offset,
        # end_lineno and end_col_offset must be copied explicitly. ast.fix_missing_locations
        # would fill them with the first node's values (line 1), making the range
        # (lineno=N, end_lineno=1) invalid and causing a ValueError at compile time.
        end_lineno=last.end_lineno,
        end_col_offset=last.end_col_offset,
    )
    # Fill in any remaining missing location fields on child nodes (e.g. the
    # ast.Name target we created above).
    ast.fix_missing_locations(tree)
    return tree, True


def compile_for_repl(code: str, tree: ast.AST) -> types.CodeType:
    """
    Register source lines in linecache and compile the AST tagged with "<repl>".

    Python's traceback formatter calls linecache.getlines(filename) to look up
    source lines when rendering a traceback. Without registration, it finds nothing
    for "<repl>" and omits the source line and ^^^ underline from the output.

    By registering the source here and tagging the compiled code object with the
    same filename, any exception during exec() produces a human-readable traceback:

        File "<repl>", line N, in <module>
          <actual source line>
          ^^^^^^^^^^^^^^^^^^^

    Args:
        code: The original source string — registered as-is so line numbers match.
        tree: The (possibly rewritten by rewrite_last_expr) AST to compile.

    Returns:
        A code object ready to pass to exec().
    """
    # linecache.cache format expected by the traceback formatter:
    # (size, mtime, lines, fullname)
    # mtime=None tells linecache never to expire this entry.
    linecache.cache["<repl>"] = (
        len(code),
        None,
        code.splitlines(keepends=True),
        "<repl>",
    )
    # Compiling with filename "<repl>" links any raised exceptions to the
    # linecache entry above, so the formatter can retrieve and display source lines.
    return compile(tree, "<repl>", "exec")
