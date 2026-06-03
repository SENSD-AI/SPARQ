import unittest

from sparq.tools.python_repl.python_repl_tool import make_python_repl_tool
from sparq.tools.python_repl.namespace import get_ns_path, cleanup_ns


class TestPythonReplTool(unittest.TestCase):

    def setUp(self):
        """Create a fresh run-scoped namespace and tool before each test."""
        self.ns_path = get_ns_path("test")
        self.tool = make_python_repl_tool(self.ns_path)

    def tearDown(self):
        cleanup_ns("test")

    def test_simple_execution(self):
        """Test basic code execution returns success message."""
        code = "x = 5\ny = 10\nprint(x + y)"
        response = self.tool.invoke({"code": code})
        self.assertIn("✓ Code executed successfully", response)
        self.assertIn("15", response)

    def test_execution_error(self):
        """Test that type errors are reported as failures."""
        code = "x = 5\ny = '10'\nprint(x + y)"
        response = self.tool.invoke({"code": code})
        self.assertIn("✗ Execution failed", response)

    def test_namespace_persistence(self):
        """Test that variables persist across executions."""
        self.tool.invoke({"code": "x = 42", "persist_namespace": True})
        response = self.tool.invoke({"code": "print(x)", "persist_namespace": True})
        self.assertIn("✓ Code executed successfully", response)
        self.assertIn("42", response)

    # def test_timeout(self):
    #     """Test that long-running code is terminated."""
    #     code = "import time\ntime.sleep(3*60)"
    #     response = self.tool.invoke({"code": code})
    #     self.assertIn("✗ Execution failed", response)
    #     self.assertIn("TimeoutError", response)


if __name__ == "__main__":
    unittest.main()
