import unittest

from sparq.tools.python_repl.python_repl_tool import python_repl_tool
from sparq.tools.python_repl.namespace import clear_persistent_namespace


class TestPythonReplTool(unittest.TestCase):

    def setUp(self):
        """Clear persistent namespace before each test."""
        clear_persistent_namespace()

    def test_simple_execution(self):
        """Test basic code execution returns success message."""
        code = "x = 5\ny = 10\nprint(x + y)"
        response = python_repl_tool.invoke({"code": code})
        self.assertIn("✓ Code executed successfully", response)
        self.assertIn("15", response)

    def test_execution_error(self):
        """Test that type errors are reported as failures."""
        code = "x = 5\ny = '10'\nprint(x + y)"
        response = python_repl_tool.invoke({"code": code})
        self.assertIn("✗ Execution failed", response)

    def test_namespace_persistence(self):
        """Test that variables persist across executions."""
        python_repl_tool.invoke({"code": "x = 42", "persist_namespace": True})
        response = python_repl_tool.invoke({"code": "print(x)", "persist_namespace": True})
        self.assertIn("✓ Code executed successfully", response)
        self.assertIn("42", response)

    # def test_timeout(self):
    #     """Test that long-running code is terminated."""
    #     code = "import time\ntime.sleep(3*60)"
    #     response = python_repl_tool.invoke({"code": code})
    #     self.assertIn("✗ Execution failed", response)
    #     self.assertIn("TimeoutError", response)


if __name__ == "__main__":
    unittest.main()
