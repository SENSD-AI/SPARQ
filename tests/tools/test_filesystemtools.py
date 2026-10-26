import unittest
from pathlib import Path

from src.sparq.tools.filesystemtools import filesystemtools


class TestFilesystemTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set working directory for tests and create it if it doesn't exist
        """
        cls.working_dir = Path("~/tmp/sparq/test_filesystemtools").expanduser()
        cls.working_dir.mkdir(parents=True, exist_ok=True)

    def test_filesystemtools(self):
        
        # Test with all tools
        tools = filesystemtools(working_dir=str(self.working_dir), selected_tools='all')
        self.assertIsNotNone(tools)
        
        # Test with specific tools
        tools = filesystemtools(working_dir=str(self.working_dir), selected_tools=['write_file', 'read_file'])
        self.assertIsNotNone(tools)
        
        # Test with invalid tool
        with self.assertRaises(ValueError):
            filesystemtools(working_dir=str(self.working_dir), selected_tools=['invalid_tool'])

    def test_write_read_txt(self):
        """
        Test the write_file tool and read_file tool
        """
        tools = filesystemtools(working_dir=str(self.working_dir), selected_tools=['write_file', 'read_file'])
        write_tool = next(tool for tool in tools if tool.name == 'write_file')
        read_tool = next(tool for tool in tools if tool.name == 'read_file')

        # Write to a file
        test_file_path = self.working_dir / "test.txt"
        write_tool.invoke({"file_path": str(test_file_path),
                           "text": "Hello, world!",
                          "append": False})

        # Read from the file
        content = read_tool.run(str(test_file_path))
        self.assertEqual(content, "Hello, world!")


