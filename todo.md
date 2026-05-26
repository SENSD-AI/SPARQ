# Incorporating v2
- [ ] Change `AgenticSystemSettings` to `BaseAgenticSettings`
    - [ ] Fix imports:
        - [ ] `tests/test_settings.py`
    - [ ] Change usage pattern of `LLMSettings`. This should be supplied by the user when a specific architecture is selected. Do: `List[LLMSetting] = Field(None, description="LLM settings for the agentic system. This should be supplied by the user when a specific architecture is selected.")`

- [ ] keep a default config for each architecture.
- [ ] `__main__.py` should receive CLI arg that specifies the architecture.
- [ ] Add a setup node that does the same thing as `setup.py`. 