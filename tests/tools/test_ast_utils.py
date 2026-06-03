import ast
import linecache
import unittest

from sparq.tools.python_repl.ast_utils import rewrite_last_expr, compile_for_repl


class TestRewriteLastExpr(unittest.TestCase):
    def test_bare_expression_is_rewritten(self):
        tree = ast.parse("a = 1\nb = 2\na + b")
        tree, has_result = rewrite_last_expr(tree)
        self.assertTrue(has_result)
        last = tree.body[-1]
        self.assertIsInstance(last, ast.Assign)
        self.assertEqual(last.targets[0].id, "__repl_result__")

    def test_no_rewrite_when_last_is_assignment(self):
        tree = ast.parse("a = 1\nb = 2")
        tree, has_result = rewrite_last_expr(tree)
        self.assertFalse(has_result)
        self.assertIsInstance(tree.body[-1], ast.Assign)
        self.assertNotEqual(tree.body[-1].targets[0].id, "__repl_result__")

    def test_single_bare_expression(self):
        tree = ast.parse("a + b")
        tree, has_result = rewrite_last_expr(tree)
        self.assertTrue(has_result)

    def test_function_definition_not_rewritten(self):
        tree = ast.parse("def foo():\n    return 42")
        tree, has_result = rewrite_last_expr(tree)
        self.assertFalse(has_result)

    def test_empty_body_not_rewritten(self):
        tree = ast.parse("")
        tree, has_result = rewrite_last_expr(tree)
        self.assertFalse(has_result)

    def test_location_info_preserved(self):
        # Ensures end_lineno is correctly copied so compile() doesn't raise ValueError
        tree = ast.parse("x = 1\nx + 1")
        tree, _ = rewrite_last_expr(tree)
        last = tree.body[-1]
        self.assertGreaterEqual(last.end_lineno, last.lineno)


class TestCompileForRepl(unittest.TestCase):
    def test_registers_in_linecache(self):
        code = "x = 1\nx + 1"
        tree = ast.parse(code)
        compile_for_repl(code, tree)
        self.assertIn("<repl>", linecache.cache)
        cached_lines = linecache.cache["<repl>"][2]
        self.assertEqual("".join(cached_lines), code)

    def test_returns_code_object(self):
        import types
        code = "x = 1"
        tree = ast.parse(code)
        result = compile_for_repl(code, tree)
        self.assertIsInstance(result, types.CodeType)

    def test_compiled_filename_is_repl(self):
        code = "x = 1"
        tree = ast.parse(code)
        code_obj = compile_for_repl(code, tree)
        self.assertEqual(code_obj.co_filename, "<repl>")

    def test_traceback_shows_source_line(self):
        # End-to-end: error in exec() should show the source line in the traceback
        import traceback as tb
        code = "x = [1, 2, 3]\nfor item in x.sort():\n    pass"
        tree = ast.parse(code)
        tree, _ = rewrite_last_expr(tree)
        code_obj = compile_for_repl(code, tree)
        try:
            exec(code_obj, {})
        except TypeError:
            formatted = tb.format_exc()
            self.assertIn("<repl>", formatted)
            self.assertIn("x.sort()", formatted)


if __name__ == "__main__":
    unittest.main()
