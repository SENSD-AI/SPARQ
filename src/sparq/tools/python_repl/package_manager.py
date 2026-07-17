"""Package manager for safe code execution with whitelisted packages."""

from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional
import tomllib
from threading import Lock


class PackageManager:
    """Manages package installation and whitelisting for safe code execution."""
    lock = Lock()

    _config = None

    DEFAULT_PACKAGE_LIST = {
        "blocked": [ "os", "sys", "subprocess", "shutil", "socket", "multiprocessing", "threading", "ctypes", "signal" ],
        "safe": [ "math", "json", "re", "datetime", "functools", "itertools", "collections", "random", "string", "time", "statistics" ],
        "whitelisted": [ "numpy", "pandas", "statsmodels", "scipy", "matplotlib", "seaborn", "plotly" ],
    }

    @staticmethod
    def _get_pip_command() -> list[str]:
        """
        Get the appropriate pip command for the current Python environment.

        Prefers uv if available, falls back to pip.

        :return: Command list to invoke pip.
        :rtype: list[str]
        """
        if shutil.which("uv"):
            return ["uv", "pip"]

        return [sys.executable, "-m", "pip"]

    @classmethod
    def load_package_config(cls, config_path: Optional[str] = None) -> dict:
        """
        Load package configuration from a TOML file.

        Args:
            config_path (Optional[str]): Path to the configuration file (In TOML format). If None, uses default paths
        Returns:
            dict: Package configuration with 'blocked', 'safe', and 'whitelisted' lists.
        """

        # Return cached config if already loaded
        if cls._config is not None:
            return cls._config

        # Define the hierarchy of config file locations
        # 1. User-specified path
        # 2. User config directory (~/.config/sparq/package_config.toml)
        # 3. Default config in the package directory
        heirarchy_paths = [
            Path(config_path) if config_path else None,
            Path.home() / ".config" / "sparq" / "package_config.toml",
            Path(__file__).parent / "package_config.toml",
        ]

        # Find the first existing config file in the heirarchy
        config_file = None
        for path in heirarchy_paths:
            if path and path.exists():
                config_file = path
                break

        try:
            with open(config_file, "rb") as f:
                config = tomllib.load(f)

                cls._config = {
                    "blocked": config["stdlib"]["blocked"],
                    "safe": config["stdlib"]["safe"],
                    "whitelisted": config["third_party"]["whitelisted"],
                }
                return cls._config
        except Exception as e:
            print(f"Error loading package config: {e}. \nUsing default package list.\n")
            cls._config = cls.DEFAULT_PACKAGE_LIST
            return cls._config

    @classmethod
    def is_whitelisted(cls, package_name: str) -> bool:
        """
        Check if a package is whitelisted for installation.

        :param package_name: Name of the package to check.
        :type package_name: str
        :return: True if whitelisted, False otherwise.
        :rtype: bool
        """
        package_config = cls.load_package_config()
        if package_name in package_config["blocked"]:
            return False
        
        return (package_name in package_config["safe"] or package_name in package_config["whitelisted"])
    
    @classmethod
    def is_installed(cls, package_name: str) -> bool:
        """
        Check if a package is actually installed.

        :param package_name: Name of the package to check.
        :type package_name: str
        :return: True if installed, False otherwise.
        :rtype: bool
        """
        import importlib
        import importlib.util
        importlib.invalidate_caches()
        return importlib.util.find_spec(package_name) is not None

    @classmethod
    def install_package(cls, package_name: str) -> dict:
        """
        Install a package if it's whitelisted.

        :param package_name: Name of the package to install.
        :type package_name: str
        :return: Result dictionary with success status and message.
        :rtype: dict
        """
        if not cls.is_whitelisted(package_name):
            print(f"Package '{package_name}' is not whitelisted for installation.")
            return {
                "success": False,
                "message": f"Package '{package_name}' is not whitelisted for installation.",
            }

        with cls.lock:
            if cls.is_installed(package_name):
                print(f"Package '{package_name}' is already installed.")
                return {
                    "success": True,
                    "message": f"Package '{package_name}' is already installed.",
                }

            try:
                cmd = cls._get_pip_command() + ["install", package_name]
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return {
                    "success": True,
                    "message": f"Package '{package_name}' installed successfully.",
                }
            except subprocess.CalledProcessError as e:
                error_detail = e.stderr.strip() if e.stderr else str(e)
                print(f"Failed to install package '{package_name}': {error_detail}")
                return {
                    "success": False,
                    "message": f"Failed to install package '{package_name}': {error_detail}",
                }

    @classmethod
    def uninstall_package(cls, package_name: str) -> dict:
        """
        Uninstall a package.

        :param package_name: Name of the package to uninstall.
        :type package_name: str
        :return: Result dictionary with success status and message.
        :rtype: dict
        """
        # If not installed, nothing to do
        if not cls.is_installed(package_name):
            return {
                "success": True,
                "message": f"Package '{package_name}' is not installed.",
            }

        try:
            cmd = cls._get_pip_command() + ["uninstall", package_name]
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            #Clear stale module cache entries
            for key in list(sys.modules.keys()):
                if key == package_name or key.startswith(f"{package_name}."):
                    del sys.modules[key]

            return {
                "success": True,
                "message": f"Package '{package_name}' uninstalled successfully.",
            }
        except subprocess.CalledProcessError as e:
            error_detail = e.stderr.strip() if e.stderr else str(e)
            return {
                "success": False,
                "message": f"Failed to uninstall package '{package_name}': {error_detail}",
            }


class PackageUtils(PackageManager):
    """Utility class for package management."""

    @classmethod
    def extract_package_name_error(cls, error_message: str) -> str | None:
        """
        Extract the package name from an ImportError message.

        :param error_message: The ImportError message.
        :type error_message: str
        :return: Extracted package name or None if not found.
        :rtype: str | None
        """
        package_config = cls.load_package_config()
        for pkg in package_config["whitelisted"] + package_config["safe"]:
            if f"No module named '{pkg}'" in error_message:
                return pkg
        return None


