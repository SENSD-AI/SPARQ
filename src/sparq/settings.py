"""
Settings.py

This module implements app configuration. Specifically, it provides the following features:
1. Path specifications: Defines paths for configuration files, environment variables, data input and data output directories.
2. Environment variable loading: Loads environment variables from a specified .env file.
3. Configuration building: Builds a configuration object.
"""

from pathlib import Path
from typing import Optional, Literal
import platform

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from sparq.utils.get_package_dir import get_project_root, get_package_dir


# -----------------------------------------------------------------------
# Configuration directories based on platform
# -----------------------------------------------------------------------
WINDOWS_CONFIG_DIR: Path = Path.home() / "AppData" / "Local" / "sparq"
DARWIN_CONFIG_DIR: Path = Path.home() / ".config" / "sparq"
LINUX_CONFIG_DIR: Path = Path.home() / ".config" / "sparq"

PLATFORM = platform.system()


def get_user_config_dir() -> Path:
    if PLATFORM == "Windows":
        return WINDOWS_CONFIG_DIR
    elif PLATFORM == "Darwin":
        return DARWIN_CONFIG_DIR
    elif PLATFORM == "Linux":
        return LINUX_CONFIG_DIR
    else:
        raise RuntimeError(f"Unsupported platform: {PLATFORM}")


INNER_CONFIG_PATH: Path = get_package_dir() / "default_config.toml"
DEV_CONFIG_PATH: Path = get_project_root() / "config.toml"
USER_CONFIG_PATH: Path = get_user_config_dir() / "config.toml"

# -----------------------------------------------------------------------
# DATA MANIFEST AND SUMMARY PATHS
# -----------------------------------------------------------------------
# TODO: Move data into package data directory
BUNDLED_DATA_DIR: Path = get_package_dir() / "data"         # Data will be copied from here to user data dir if not present
USER_DATA_DIR: Path = get_user_config_dir() / "data"

DATA_MANIFEST_PATH: Path = USER_DATA_DIR / "data_manifest.json"
DATA_SUMMARIES_PATH: Path = USER_DATA_DIR / "data_summaries.json"
DATA_SUMMARIES_FULL_PATH: Path = USER_DATA_DIR / "data_summaries_full.json"
DATA_SUMMARIES_SHORT_PATH: Path = USER_DATA_DIR / "data_summaries_short.json"

# -----------------------------------------------------------------------
# dotfile paths
# -----------------------------------------------------------------------
DEV_DOTFILE_PATH: Path = get_project_root() / ".env"
USER_DOTFILE_PATH: Path = get_user_config_dir() / ".env"


# -----------------------------------------------------------------------
# Environment settings
# -----------------------------------------------------------------------
class ENVSettings(BaseSettings):
    # Load environment variables (sequence: development .env, user .env)
    model_config = SettingsConfigDict(
        env_file=(DEV_DOTFILE_PATH, USER_DOTFILE_PATH), extra="ignore"
    )

    # API Keys and Access Tokens
    google_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    hf_token: Optional[str] = None

    # Optional AWS settings
    aws_profile: Optional[str] = None
    aws_region: Optional[str] = None


# -----------------------------------------------------------------------
# Agentic system settings (llms, output paths, prompt paths)
# -----------------------------------------------------------------------
class LLMSetting(BaseModel):
    provider: str
    model_name: str = Field(..., alias="model")
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning: Optional[Literal[1, 0]] = None


class LLMSettings(BaseModel):
    router: Optional[LLMSetting] = None
    planner: Optional[LLMSetting] = None
    executor: Optional[LLMSetting] = None
    aggregator: Optional[LLMSetting] = None


class PathSettings(BaseModel):
    prompts_dir: Path
    output_dir: Path
    data_manifest_path: Path
    data_summary_path: Path


class AgenticSystemSettings(BaseSettings):
    test_query: str  # In Inner config file
    llm_config: LLMSettings
    paths: PathSettings

    model_config = SettingsConfigDict(
        toml_file=[INNER_CONFIG_PATH, DEV_CONFIG_PATH, USER_CONFIG_PATH]
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls, deep_merge=True),)
