"""
setup.py

This module sets up the environment, llm settings, data paths and downloads the required
data (data used by SPARQ).

1. Creates user config and data directories if they do not already exist.
2. Download Data from huggingface       # FUTURE NOTE: Data will be moved to a different database later.
3. Copies bundled data to the user data directory if it does not already exist.
4. Writes a sentinel file to mark setup as complete.

"""
import shutil

from sparq.settings import (
    BUNDLED_DATA_DIR,
    USER_DATA_DIR,
    USER_DOTFILE_PATH,
    DATA_SUMMARIES_PATH,
    DATA_SUMMARIES_FULL_PATH,
    DATA_SUMMARIES_SHORT_PATH,
    get_user_config_dir,
)
from sparq.architectures.v1.settings import V1_INNER_CONFIG, V1_USER_CONFIG

SENTINEL = get_user_config_dir() / ".setup_complete"

ARCH_CONFIGS = [
    (V1_INNER_CONFIG, V1_USER_CONFIG),
    # (V2_INNER_CONFIG, V2_USER_CONFIG),  # add when v2 is implemented
]


def setup():
    # 1. Create user config and data directories
    get_user_config_dir().mkdir(parents=True, exist_ok=True)
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Copy each architecture's default config to its user config dir
    for inner, user in ARCH_CONFIGS:
        user.parent.mkdir(parents=True, exist_ok=True)
        if not user.exists():
            shutil.copy2(inner, user)

    # 4. Copy .env.example to user config dir if no .env exists
    if not USER_DOTFILE_PATH.exists():
        shutil.copy2(BUNDLED_DATA_DIR.parent / ".env.example", USER_DOTFILE_PATH)
        print(f"Template .env created at {USER_DOTFILE_PATH} — please fill in your API keys.")

    # 5. Copy bundled data summaries to user data dir if not already present
    for src, dst in [
        (BUNDLED_DATA_DIR / "data_summaries.json",       DATA_SUMMARIES_PATH),
        (BUNDLED_DATA_DIR / "data_summaries_full.json",  DATA_SUMMARIES_FULL_PATH),
        (BUNDLED_DATA_DIR / "data_summaries_short.json", DATA_SUMMARIES_SHORT_PATH),
    ]:
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # 6. Download datasets from HuggingFace
    from sparq.utils.download_data import main as download_data
    try:
        download_data()
    except ValueError as e:
        print(f"Warning: Data download skipped — {e}")

    # 7. Write sentinel to mark setup as complete
    SENTINEL.write_text("")
