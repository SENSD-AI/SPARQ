import os
from typing import Any

from langchain_core.language_models import BaseChatModel


def _make_native(model: str, provider: str, **options: Any) -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    if provider == "openai" and "top_k" in options:
        raise ValueError("top_k is not supported by the openai provider")
    return init_chat_model(model=model, model_provider=provider, **options)


def _make_openrouter(model: str, provider: str, **options: Any) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")

    base_url = os.getenv("OPENROUTER_BASE_URL")
    if not base_url:
        raise ValueError("OPENROUTER_BASE_URL not found")

    top_k = options.pop("top_k", None)
    if top_k is not None:
        options["extra_body"] = {"top_k": top_k}

    return ChatOpenAI(
        api_key=SecretStr(api_key),
        base_url=base_url,
        model=model or "meta-llama/llama-4-maverick:free",
        **options,
    )


def _make_bedrock(model: str, provider: str, **options: Any) -> BaseChatModel:
    import boto3
    from langchain_aws import ChatBedrockConverse

    # Kept for interface consistency with other factories.
    _ = provider

    region = os.getenv("AWS_REGION")
    profile = os.getenv("AWS_PROFILE")

    missing = []
    if not region:
        missing.append("AWS_REGION")
    if not profile:
        missing.append("AWS_PROFILE")
    if missing:
        raise ValueError(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )

    top_k = options.pop("top_k", None)
    if top_k is not None:
        options["additional_model_request_fields"] = {"top_k": top_k}

    try:
        from botocore.config import Config as BotocoreConfig

        session = boto3.Session(profile_name=profile, region_name=region)
        client = session.client(
            service_name="bedrock-runtime",
            config=BotocoreConfig(
                connect_timeout=10,
                read_timeout=300,
                retries={
                    "mode": "standard",
                    "max_attempts": 3,
                },  # exponential backoff on transient errors incl. ReadTimeoutError
            ),  # https://docs.aws.amazon.com/botocore/latest/reference/config.html
        )
        return ChatBedrockConverse(model=model, client=client, **options)
    except Exception as exc:
        raise ValueError(
            f"Failed to initialize Bedrock model '{model}' "
            f"(profile='{profile}', region='{region}')."
        ) from exc


_PROVIDER_FACTORIES = {
    "openai": _make_native,
    "google_genai": _make_native,
    "openrouter": _make_openrouter,
    "aws_bedrock": _make_bedrock,
}


def get_llm(
    model: str = "gpt-4o",
    provider: str = "openai",
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> BaseChatModel:
    """Construct a model, forwarding generation options to its provider.

    Unset options are omitted. SPARQ's defaults are resolved by its configuration
    before calling this function. Sampling support depends on the selected model.
    """
    factory = _PROVIDER_FACTORIES.get(provider)
    if factory is None:
        supported = ", ".join(f"'{p}'" for p in _PROVIDER_FACTORIES)
        raise ValueError(
            f"Provider '{provider}' not supported. Choose from: {supported}."
        )
    options: dict[str, Any] = {}
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if top_k is not None:
        options["top_k"] = top_k

    return factory(model, provider, **options)
