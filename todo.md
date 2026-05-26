# Incorporating v2
- [x] Change `AgenticSystemSettings` to `BaseAgenticSettings`
    - [x] Fix imports:
        - [x] `tests/test_settings.py`
    - [x] Change usage pattern of `LLMSettings`. Introduced `BaseLLMSettings` + `BaseAgenticSettings[LLMConfigT: BaseLLMSettings]` generic pattern.

- [x] keep a default config for each architecture.
    - [x] the config file for each architecture should be copied into `USER_CONFIG_DIR/<architecture>/config.toml` on first run (done via `setup.py`).
- [ ] `__main__.py` should receive CLI arg that specifies the architecture.
- [ ] Add a setup node that does the same thing as `setup.py`. 