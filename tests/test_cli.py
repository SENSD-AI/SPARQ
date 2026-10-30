"""Check that unsupported architectures fail before model initialization."""

import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from sparq.__main__ import main


class TestCLI(unittest.TestCase):
    def test_unimplemented_v2_is_rejected(self):
        errors = io.StringIO()
        with (
            patch("sys.argv", ["sparq", "--architecture", "v2"]),
            patch("sparq.__main__.ENVSettings") as environment,
            redirect_stderr(errors),
            self.assertRaises(SystemExit) as raised,
        ):
            main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid choice", errors.getvalue())
        environment.assert_not_called()
