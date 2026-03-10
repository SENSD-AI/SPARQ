import os


def _make_native(model: str, provider: str):
    from langchain.chat_models import init_chat_model
    return init_chat_model(model=model, model_provider=provider)


def _make_openrouter(model: str, provider: str):
    from langchain_openai import ChatOpenAI

    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found")

    base_url = os.getenv('OPENROUTER_BASE_URL')
    if not base_url:
        raise ValueError("OPENROUTER_BASE_URL not found")

    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=base_url,
        model_name=model or "meta-llama/llama-4-maverick:free",
    )

def _make_bedrock(model: str, provider: str):
    from langchain_aws import ChatBedrock
    import boto3

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
        raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")

    try:
        session = boto3.Session(profile_name=profile, region_name=region)
        client = session.client(service_name="bedrock-runtime")
        return ChatBedrock(model=model, client=client)
    except Exception as exc:
        raise ValueError(
            f"Failed to initialize Bedrock model '{model}' "
            f"(profile='{profile}', region='{region}')."
        ) from exc


_PROVIDER_FACTORIES = {
    'openai':       _make_native,
    'google_genai': _make_native,
    'openrouter':   _make_openrouter,
    'aws_bedrock':  _make_bedrock,
}


def get_llm(model: str = 'gpt-4o', provider: str = 'openai'):
    factory = _PROVIDER_FACTORIES.get(provider)
    if factory is None:
        supported = ', '.join(f"'{p}'" for p in _PROVIDER_FACTORIES)
        raise ValueError(f"Provider '{provider}' not supported. Choose from: {supported}.")
    return factory(model, provider)