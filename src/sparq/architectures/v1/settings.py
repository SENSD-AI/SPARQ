from pathlib import Path
from typing import Optional

from pydantic_settings import SettingsConfigDict

from sparq.settings import BaseAgenticSettings, BaseLLMSettings, LLMSetting, get_user_config_dir
from sparq.utils.get_package_dir import get_project_root

V1_INNER_CONFIG: Path = Path(__file__).parent / "default_config.toml"
V1_DEV_CONFIG:   Path = get_project_root() / "config_v1.toml"
V1_USER_CONFIG:  Path = get_user_config_dir() / "v1" / "config.toml"


class V1LLMSettings(BaseLLMSettings):
    router:     Optional[LLMSetting] = None
    planner:    Optional[LLMSetting] = None
    executor:   Optional[LLMSetting] = None
    aggregator: Optional[LLMSetting] = None


class V1Settings(BaseAgenticSettings[V1LLMSettings]):
    model_config = SettingsConfigDict(
        toml_file=[V1_INNER_CONFIG, V1_DEV_CONFIG, V1_USER_CONFIG]
    )
