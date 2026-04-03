"""
Settings.py

This module implements app configuration. Specifically, it provides the following features:
1. Path specifications: Defines paths for configuration files, environment variables, data input and data output directories.
2. Environment variable loading: Loads environment variables from a specified .env file.
3. Configuration building: Builds a configuration object.
"""

from pathlib import Path
from typing import Annotated, Any, Optional, Literal, Self
import platform
import os

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
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
# Data Manifest and Summaries
# -----------------------------------------------------------------------
BUNDLED_DATA_DIR: Path = get_package_dir() / "data"         # On first install, Data will be copied from here to user data dir if not present
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
# Other Defaults
# -----------------------------------------------------------------------
DEFAULT_RECURSION_LIMIT = 20  # Default maximum number of steps it takes to walk from graph root to terminal leaf


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
    langsmith_tracing: Optional[str] = None
    langsmith_project: Optional[str] = None
    hf_token: Optional[str] = None

    # Optional AWS settings
    aws_profile: Optional[str] = None
    aws_region: Optional[str] = None

    def __init__(self, verbose: bool = False, **data):
        super().__init__(**data)
        if verbose:
            print(self.model_dump_json(indent=2))

    def model_post_init(self, _) -> None:
        for key, val in self.model_dump().items():
            if val is not None:
                os.environ[key.upper()] = str(val)


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
    recursion_limit: int = Field(DEFAULT_RECURSION_LIMIT, ge=0, le=200, description="Maximum number of steps it takes to walk from graph root to terminal leaf")


class LLMSettings(BaseModel):
    router: Optional[LLMSetting] = None
    planner: Optional[LLMSetting] = None
    executor: Optional[LLMSetting] = None
    aggregator: Optional[LLMSetting] = None


class PathSettings(BaseModel):
    prompts_dir: Annotated[Path, NoDecode]
    output_dir: Annotated[Path, NoDecode]
    output_stem: Optional[str] = Field(None, alias="output_stem", description="Stem for output files, e.g., setting to 'output' would give 'output_<timestamp>.json'")
    run_dir: Optional[Path] = None

    @field_validator("prompts_dir", "output_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str) -> Path:
        p = Path(v).expanduser()
        if not p.is_absolute():
            pkg_dir = get_package_dir()
            assert pkg_dir is not None, "Could not locate package directory"
            p = pkg_dir / p
        return p

    @model_validator(mode="after")
    def set_run_dir(self) -> Self:
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        stem = f"{self.output_stem}_" if self.output_stem else ""
        self.run_dir = self.output_dir / f"{stem}{timestamp}"
        return self

class AgenticSystemSettings(BaseSettings):
    test_query: str  # In Inner config file. Devs and Users aren't expected.
    llm_config: LLMSettings
    paths: PathSettings

    model_config = SettingsConfigDict(
        toml_file=[INNER_CONFIG_PATH, DEV_CONFIG_PATH, USER_CONFIG_PATH]
    )

    def __init__(self, verbose: bool = False, **data):
        super().__init__(**data)
        if verbose:
            print(self.model_dump_json(indent=2))

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
