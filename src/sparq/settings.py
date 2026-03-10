from pathlib import Path
from typing import Dict
import os
import sys
from datetime import datetime as dt

import appdirs as ad
from dotenv import load_dotenv

from sparq.utils.get_package_dir import get_package_dir, get_project_root

class Settings:
    """Default settings for SPARQ with sensible defaults.
    
    To override, create a subclass and provide your own values or
    pass in different paths during initialization.

    Values that can be overridden:
    - config_path: Path to the LLM configuration TOML file.
    - prompts_dir: Directory containing prompt text files.
    """
    def __init__(self, 
                 prompts_dir: Path = None, 
                 config_path: Path = None, 
                 env_path: Path = None):
        # Set package and project root directories
        self.PACKAGE_DIR = get_package_dir()
        self.PROJECT_ROOT = get_project_root()

        if self.PACKAGE_DIR is None or self.PROJECT_ROOT is None:
            raise RuntimeError("Could not determine package or project root directory.")
        
        # Create .config/sparq directory
        self.USER_CONFIG_DIR = self._get_user_config_dir()
        self.USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Load environment variables from .env file
        self._load_env_variables(env_path)

        # Load configuration
        self.CONFIG_PATH = config_path or (self.PACKAGE_DIR / "default_config.toml")
        self.CONFIG = self.load_config(self.CONFIG_PATH)

        self.LLM_CONFIG = self.CONFIG['llm_config']
        self.OUTPUT_DIR = Path(os.path.expanduser(self.CONFIG['paths']['OUTPUT_DIR']))
        try:
            self.OUTPUT_STEM = self.CONFIG['paths']['OUTPUT_STEM']
        except KeyError:
            self.OUTPUT_STEM = None

        self.DATA_MANIFEST_PATH = Path(os.path.expanduser(self.CONFIG['paths']['DATA_MANIFEST_PATH']))
        self.DATA_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Set output directories for different agents
        self.ROUTER_OUTPUT_DIR = self.OUTPUT_DIR / "router"
        self.PLANNER_OUTPUT_DIR = self.OUTPUT_DIR / "planner"
        self.EXECUTOR_OUTPUT_DIR = self.OUTPUT_DIR / "executor"
        self.AGGREGATOR_OUTPUT_DIR = self.OUTPUT_DIR / "aggregator"

        # Create output directories
        self._create_output_dirs()
        
        # Allow override of prompts directory (defaults to PACKAGE_DIR / CONFIG['paths']['PROMPTS_DIR'])
        # Note: Prompts directory isn't expected in ~/.config/sparq/ because these prompts are not meant to be changed by the user.
        # They are only controlled by the developer, who will clone the repository and modify the prompts as needed.
        self.PROMPTS_DIR = prompts_dir or self.PACKAGE_DIR / self.CONFIG['paths']['PROMPTS_DIR']
        self.PROMPTS_DIR = Path(self.PROMPTS_DIR)

    def _get_user_config_dir(self):
        if sys.platform in ("darwin", "linux"):
            return Path.home() / ".config" / "sparq"
        else:
            return Path(ad.user_config_dir("sparq"))
    
    def load_prompts(self) -> Dict[str, str]:
        """Load all prompts from the prompts directory."""
        from sparq.utils.helpers import load_text

        self.ROUTER_PROMPT_PATH = self.PROMPTS_DIR / "router_message.txt"
        self.PLANNER_PROMPT_PATH = self.PROMPTS_DIR / "planner_message.txt"
        self.EXPLORER_PROMPT_PATH = self.PROMPTS_DIR / "explorer_message.txt"
        self.ANALYZER_PROMPT_PATH = self.PROMPTS_DIR / "analyzer_message.txt"
        self.EXECUTOR_PROMPT_PATH = self.PROMPTS_DIR / "executor_message.txt"
        self.AGGREGATOR_PROMPT_PATH = self.PROMPTS_DIR / "aggregator_message.txt"
        
        return {
            'router_prompt': load_text(self.ROUTER_PROMPT_PATH),
            'planner_prompt': load_text(self.PLANNER_PROMPT_PATH),
            'explorer_prompt': load_text(self.EXPLORER_PROMPT_PATH),
            'analyzer_prompt': load_text(self.ANALYZER_PROMPT_PATH),
            'executor_prompt': load_text(self.EXECUTOR_PROMPT_PATH),
            'aggregator_prompt': load_text(self.AGGREGATOR_PROMPT_PATH)
        }

    def load_config(self, path_to_toml_file=None):
        """
        Takes in a toml file and returns a dictionary
        with llm models and providers
        """
        import tomllib
        
        path_to_toml_file = self.CONFIG_PATH if path_to_toml_file is None else path_to_toml_file
        with open(path_to_toml_file, mode="rb") as f:
            config = tomllib.load(f)
        
        return config
    
    def _create_output_dirs(self):
        """Creates timestamped output directories.
        
        Output structure:
        - {OUTPUT_DIR}/{stem}_{timestamp_str}/
            - router/
            - planner/
            - executor/
            - aggregator/
        
        """

        timestamp_dt: dt = dt.now()
        timestamp_str: str = timestamp_dt.strftime("%d-%m-%Y_%H-%M-%S")
        stem = "" if self.OUTPUT_STEM is None else self.OUTPUT_STEM

        dirs = [
            Path(f"{self.OUTPUT_DIR}/{stem}_{timestamp_str}"),
            self.ROUTER_OUTPUT_DIR,
            self.PLANNER_OUTPUT_DIR,
            self.EXECUTOR_OUTPUT_DIR,
            self.AGGREGATOR_OUTPUT_DIR
        ]
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)
    
    def _load_env_variables(self, env_path: Path = None):
        """Load environment variables from a .env file if it exists."""
        paths = [
            env_path,
            self.USER_CONFIG_DIR / ".env",
            self.PROJECT_ROOT / ".env",
        ]

        for path in paths:
            if path and path.is_file():
                load_dotenv(dotenv_path=path)
                return
        
        # If no .env file exists, create a template one in the user config directory to guide the user in setting up their environment variables.
        self._create_template_env_file(env_path or self.USER_CONFIG_DIR / ".env")
        
    
    def _create_template_env_file(self, env_path: Path):
        """Create a template .env file with placeholder values."""

        template = """# SPARQ .env file template
# Replace the placeholder values with your actual API keys and configurations
# GOOGLE_API_KEY=your_google_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
# LANSMITH_API_KEY=your_lansmith_api_key_here
# LANGSMITH_PROJECT_NAME=your_project_name_here
# LANGSMITH_TRACING=true
# OTHER_API_KEY=your_other_api_key_here
# """
        with open(env_path, "w") as f:
            f.write(template)
        print(f"Template .env file created at {env_path}. Please update it with your actual values.")
    

if __name__ == "__main__":
    # Test loading prompts
    settings = Settings()
    prompts = settings.load_prompts()
    print(prompts.keys())