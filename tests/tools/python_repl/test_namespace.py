import unittest

from sparq.tools.python_repl.namespace import clean_namespace


class TestCleanNamespace(unittest.TestCase):
    def test_clean_namespace_preserves_modules(self):
        namespace = {
            "__builtins__": object(),
            "__name__": "x",
            "__modules__": {"np": "numpy"},
            "user_var": 123,
        }

        clean_namespace(namespace)

        self.assertNotIn("__builtins__", namespace)
        self.assertNotIn("__name__", namespace)
        self.assertEqual(namespace["__modules__"], {"np": "numpy"})
        self.assertEqual(namespace["user_var"], 123)
