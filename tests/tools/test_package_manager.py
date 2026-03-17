import unittest
import subprocess
import sys

from sparq.tools.python_repl.executor import execute_code
from sparq.tools.python_repl.namespace import clear_persistent_namespace
from sparq.tools.python_repl.package_manager import PackageManager, PackageUtils

class TestPackageManagerSmoke(unittest.TestCase):
    """Quick smoke tests without actual installation."""
    
    def test_whitelist_check(self):
        """Test whitelist checking without installation."""
        self.assertTrue(PackageManager.is_whitelisted('numpy'))
        self.assertTrue(PackageManager.is_whitelisted('pandas'))
        self.assertTrue(PackageManager.is_whitelisted('math'))
        self.assertFalse(PackageManager.is_whitelisted('os'))
        self.assertFalse(PackageManager.is_whitelisted('subprocess'))
    
    def test_blocked_packages(self):
        """Test that dangerous packages are blocked."""
        dangerous = ['os', 'sys', 'subprocess', 'socket', 'multiprocessing']
        for pkg in dangerous:
            self.assertFalse(
                PackageManager.is_whitelisted(pkg),
                f"Dangerous package '{pkg}' should be blocked"
            )
    
    def test_safe_stdlib_packages(self):
        """Test that safe stdlib packages are allowed."""
        safe = ['math', 'json', 're', 'datetime', 'random']
        for pkg in safe:
            self.assertTrue(
                PackageManager.is_whitelisted(pkg),
                f"Safe package '{pkg}' should be whitelisted"
            )
    
    def test_extract_package_name_formats(self):
        """Test package name extraction from various error formats."""
        test_cases = [
            ("No module named 'numpy'", 'numpy'),
            ("No module named 'pandas'", 'pandas'),
            ("No module named 'scipy'", 'scipy'),
            ("No module named 'fake_package'", None),  # Not whitelisted
        ]
        
        for error_msg, expected in test_cases:
            result = PackageUtils.extract_package_name_error(error_msg)
            self.assertEqual(
                result, expected,
                f"Failed to extract '{expected}' from '{error_msg}'"
            )

class TestPackageInstallation(unittest.TestCase):
    
    def setUp(self):
        """Clear persistent namespace before each test."""
        clear_persistent_namespace()

    @classmethod
    def setUpClass(cls):
        """Prepare test environment: uninstall test packages before all tests."""
        # Uninstall packages that might interfere with tests
        test_packages = ['numpy', 'pandas']
        for package in test_packages:
            PackageManager.uninstall_package(package)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up: uninstall test packages after all tests."""
        # Uninstall packages that were installed during testing
        test_packages = ['numpy', 'pandas']
        for package in test_packages:
            PackageManager.uninstall_package(package)

    # Naming tests with 'aaa_' to ensure they run first
    def test_aaa_whitelisted_package_auto_install(self):
        """Test that whitelisted packages are auto-installed on ImportError."""
        # Skip if numpy already installed (can't test auto-install without fresh process)
        if PackageManager.is_installed('numpy'):
            self.skipTest("numpy already installed - cannot test auto-install in same process")
        
        # Try to import numpy (should trigger auto-install)
        code = """
import numpy as np
arr = np.array([1, 2, 3])
arr.mean()
"""
        result = execute_code(code, persist_namespace=True, timeout=30)
        
        # Should succeed after auto-installation
        self.assertTrue(result.success, f"Failed: {result.error}")
        self.assertEqual(result.output, "2.0")
        
        # Verify numpy is now installed
        self.assertTrue(PackageManager.is_installed('numpy'))
    
    def test_non_whitelisted_package_fails(self):
        """Test that non-whitelisted packages are NOT auto-installed."""
        # Try to import a non-whitelisted package
        code = "import some_random_nonexistent_package"
        result = execute_code(code, persist_namespace=False, timeout=10)
        
        # Should fail with ModuleNotFoundError
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.type, "ModuleNotFoundError")
        
        # Should indicate package install failed
        if "package_install_failed" in result.error.extra_context:
            self.assertIn("not whitelisted", result.error.extra_context["package_install_failed"]["message"])
    
    def test_aac_package_persistence_after_install(self):
        """Test that installed packages persist across executions."""
        
        # First execution: Import pandas (triggers install)
        code1 = "import pandas as pd"
        result1 = execute_code(code1, persist_namespace=True, timeout=30)
        self.assertTrue(result1.success, f"Failed: {result1.error}")
        
        # Second execution: Use pandas (should work without reinstalling)
        code2 = "df = pd.DataFrame({'a': [1, 2, 3]})\ndf['a'].sum()"
        result2 = execute_code(code2, persist_namespace=True, timeout=10)
        self.assertTrue(result2.success, f"Failed: {result2.error}")
        self.assertEqual(result2.output, "6")
        
    def test_is_whitelisted(self):
        """Test PackageUtils.is_whitelisted method."""
        # Test whitelisted packages
        self.assertTrue(PackageManager.is_whitelisted('numpy'))
        self.assertTrue(PackageUtils.is_whitelisted('pandas'))
        self.assertTrue(PackageUtils.is_whitelisted('scipy'))
        self.assertTrue(PackageUtils.is_whitelisted('matplotlib'))
        
        # Test safe stdlib packages
        self.assertTrue(PackageUtils.is_whitelisted('math'))
        self.assertTrue(PackageUtils.is_whitelisted('json'))
        self.assertTrue(PackageUtils.is_whitelisted('re'))
        
        # Test blocked packages
        self.assertFalse(PackageUtils.is_whitelisted('os'))
        self.assertFalse(PackageUtils.is_whitelisted('sys'))
        self.assertFalse(PackageUtils.is_whitelisted('subprocess'))
        
        # Test non-existent packages
        self.assertFalse(PackageUtils.is_whitelisted('random_fake_package'))
    
    def test_is_installed(self):
        """Test PackageUtils.is_installed method."""
        # Test with a package that should be installed (json is stdlib)
        self.assertTrue(PackageUtils.is_installed('json'))
        
        # Test with a package that might not be installed
        if PackageUtils.is_installed('numpy'):
            self.assertTrue(PackageUtils.is_installed('numpy'))
        
        # Test with definitely non-existent package
        self.assertFalse(PackageUtils.is_installed('definitely_not_a_real_package_12345'))
    
    def test_extract_package_name_from_error(self):
        """Test extracting package name from ModuleNotFoundError message."""
        # Test various error message formats
        error_msg1 = "No module named 'numpy'"
        self.assertEqual(PackageUtils.extract_package_name_error(error_msg1), 'numpy')
        
        error_msg2 = "No module named 'pandas'"
        self.assertEqual(PackageUtils.extract_package_name_error(error_msg2), 'pandas')
        
        # Test with non-whitelisted package (should return None)
        error_msg3 = "No module named 'some_random_package'"
        result = PackageUtils.extract_package_name_error(error_msg3)
        # Only returns package name if it's whitelisted
        self.assertIsNone(result)
    
    def test_aab_multiple_package_imports(self):
        """Test importing multiple packages in one execution."""
        
        code = """
import numpy as np
import pandas as pd

arr = np.array([1, 2, 3])
df = pd.DataFrame({'col': [4, 5, 6]})

arr.sum() + df['col'].sum()
"""
        result = execute_code(code, persist_namespace=True, timeout=60)

        self.assertTrue(result.success, f"Failed: {result.error}")
        self.assertEqual(result.output, "21")  # 6 + 15
    
    def test_uninstall_package(self):
        """Test uninstalling a package"""
        package = 'numpy'

        # Ensure package is installed
        PackageManager.install_package(package)
        
        # Now uninstall
        result = PackageManager.uninstall_package(package)
        self.assertTrue(result['success'])
        self.assertIn("uninstalled successfully", result['message'])
    
    def test_install_package(self):
        """Test Installing a package"""
        # Uninstall first
        PackageManager.uninstall_package('numpy')
        
        # Install using PackageManager
        result = PackageManager.install_package('numpy')
        
        self.assertTrue(result['success'])
        self.assertIn("installed successfully", result['message'])
        
        # Verify it's now installed
        self.assertTrue(PackageManager.is_installed('numpy'))
    
    def test_install_blocked_package_fails(self):
        """Test that blocked packages cannot be installed."""
        result = PackageManager.install_package('subprocess')
        
        self.assertFalse(result['success'])
        self.assertIn("not whitelisted", result['message'])
    


class TestPackageConfig(unittest.TestCase):
    """Test package configuration loading."""
    
    def test_load_package_config(self):
        """Test that package config loads correctly."""
        config = PackageManager.load_package_config()
        
        # Check structure
        self.assertIn('blocked', config)
        self.assertIn('safe', config)
        self.assertIn('whitelisted', config)
        
        # Check some expected values
        self.assertIsInstance(config['blocked'], list)
        self.assertIsInstance(config['safe'], list)
        self.assertIsInstance(config['whitelisted'], list)
        
        # Verify some expected packages
        self.assertIn('os', config['blocked'])
        self.assertIn('math', config['safe'])
        self.assertIn('numpy', config['whitelisted'])
    
    def test_config_caching(self):
        """Test that config is cached after first load."""
        # Load config twice
        config1 = PackageManager.load_package_config()
        config2 = PackageManager.load_package_config()
        
        # Should be the same object (cached)
        self.assertIs(config1, config2)


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)