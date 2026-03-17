import unittest

from sparq.tools.python_repl.executor import execute_code
from sparq.tools.python_repl.namespace import clear_persistent_namespace


class TestExecutor(unittest.TestCase):

    def setUp(self):
        """Clear persistent namespace before each test."""
        clear_persistent_namespace()
    
    def test_basic_execution(self):
        """Test basic variable assignment and expression evaluation."""
        code = "a = 1\nb = 2\na + b"
        result = execute_code(code, persist_namespace=False, timeout=5)
        
        self.assertTrue(result.success)
        self.assertEqual(result.output, "3")
        
        # Check that new variables are in result.namespace
        self.assertIn("a", result.namespace)
        self.assertIn("b", result.namespace)
        self.assertEqual(result.namespace["a"], 1)
        self.assertEqual(result.namespace["b"], 2)

    def test_importing(self):
        """Test module import and usage."""
        # First execution: import math
        code1 = "import math"
        result1 = execute_code(code1, persist_namespace=True, timeout=5)
        
        self.assertTrue(result1.success)
        # Module should be in result.namespace["__modules__"], NOT directly in result.namespace
        self.assertIn("math", result1.namespace.get("__modules__", {}))
        self.assertEqual(result1.namespace["__modules__"]["math"], "math")
        
        # Second execution: use math (should persist)
        code2 = "math.sqrt(16)"
        result2 = execute_code(code2, persist_namespace=True, timeout=5)
        
        self.assertTrue(result2.success)
        self.assertEqual(result2.output, "4.0")
        
        # Persistence already verified by result2.success above

    def test_persistence(self):
        """Test that variables persist across executions."""
        # Execution 1: Define variable
        result1 = execute_code("x = 10", persist_namespace=True, timeout=5)
        self.assertTrue(result1.success)
        self.assertIn("x", result1.namespace)
        
        # Execution 2: Use persisted variable
        result2 = execute_code("y = x + 5", persist_namespace=True, timeout=5)
        self.assertTrue(result2.success)
        self.assertEqual(result2.output, "")  # No expression, just assignment
        self.assertIn("y", result2.namespace)
        self.assertEqual(result2.namespace["y"], 15)
        
        # Execution 3: Access both variables
        result3 = execute_code("x + y", persist_namespace=True, timeout=5)
        self.assertTrue(result3.success)
        self.assertEqual(result3.output, "25")

    def test_non_persistence(self):
        """Test that variables don't persist when persist_namespace=False."""
        # Execution 1: Define variable
        result1 = execute_code("x = 10", persist_namespace=False, timeout=5)
        self.assertTrue(result1.success)
        
        # Execution 2: Try to use variable (should fail)
        result2 = execute_code("x + 5", persist_namespace=False, timeout=5)
        self.assertFalse(result2.success)
        self.assertIsNotNone(result2.error)
        self.assertEqual(result2.error.type, "NameError")

    def test_syntax_error(self):
        """Test handling of syntax errors."""
        code = "if True"  # Missing colon
        result = execute_code(code, persist_namespace=False, timeout=5)
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.type, "SyntaxError")

    def test_runtime_error(self):
        """Test handling of runtime errors."""
        code = "1 / 0"
        result = execute_code(code, persist_namespace=False, timeout=5)
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.type, "ZeroDivisionError")

    def test_timeout(self):
        """Test that code execution times out properly."""
        code = "import time\ntime.sleep(10)"
        result = execute_code(code, persist_namespace=False, timeout=1)
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.type, "TimeoutError")

    def test_function_definition(self):
        """Test that functions are correctly identified as unpicklable."""
        code = """
def add(a, b):
    return a + b

add(3, 5)
"""
        result = execute_code(code, persist_namespace=False, timeout=5)
        
        self.assertTrue(result.success)
        self.assertEqual(result.output, "8")
        self.assertIn("add", result.namespace.get("__unpicklable__", {}))

    def test_recursive_function(self):
        """Test that recursive functions work without hanging."""
        code = """
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

fib(10)
"""
        result = execute_code(code, persist_namespace=False, timeout=5)
        
        self.assertTrue(result.success)
        self.assertEqual(result.output, "55")

    def test_unpicklable_object(self):
        """Test handling of unpicklable objects."""
        code = "f = lambda x: x * 2"                                        # Lambda functions are not picklable
        result = execute_code(code, persist_namespace=False, timeout=5)
        
        self.assertTrue(result.success)
        self.assertIn("f", result.namespace['__unpicklable__'])

    def test_print_statements(self):
        """Test that print statements are captured."""
        code = "print('Hello, World!')"
        result = execute_code(code, persist_namespace=False, timeout=5)
        
        self.assertTrue(result.success)
        self.assertEqual(result.output, "Hello, World!")

    def test_multiple_imports(self):
        """Test importing multiple modules."""
        code1 = "import math\nimport json"
        result1 = execute_code(code1, persist_namespace=True, timeout=5)
        
        self.assertTrue(result1.success)
        self.assertIn("math", result1.namespace.get("__modules__", {}))
        self.assertIn("json", result1.namespace.get("__modules__", {}))
        
        # Use both modules
        code2 = "math.pi"
        result2 = execute_code(code2, persist_namespace=True, timeout=5)
        self.assertTrue(result2.success)

    def test_saving_plots(self):
        """Test that plots can be generated and saved."""
        import os
        
        code = """import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig('test_plot.png')
"""
        result = execute_code(code, persist_namespace=False, timeout=5)
        self.assertTrue(result.success)

        # Check that the plot file was created
        self.assertTrue(os.path.exists('test_plot.png'))

        # Clean up: Delete the file after test
        if os.path.exists('test_plot.png'):
            os.remove('test_plot.png')

if __name__ == "__main__":
    unittest.main()