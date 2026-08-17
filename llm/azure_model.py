"""
Azure OpenAI model factory using LangChain.

Authentication uses a Service Principal (Client Credentials flow) via:
    AZURE_OPENAI_TENANT_ID
    AZURE_OPENAI_CLIENT_ID
    AZURE_OPENAI_CLIENT_SECRET

Connection settings:
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_VERSION
    AZURE_OPENAI_DEPLOYMENT
    AZURE_OPENAI_MODEL          (standard / default deployment)
    AZURE_OPENAI_ADVANCED_MODEL (advanced deployment)
"""

import os
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI

load_dotenv()

# Cognitive Services scope required for Azure OpenAI token auth
_AZURE_OPENAI_SCOPE = "https://cognitiveservices.azure.com/.default"


def _build_token_provider():
    """
    Build an Azure AD token provider using the Service Principal credentials
    defined in the .env file.
    """
    tenant_id = os.getenv("AZURE_OPENAI_TENANT_ID")
    client_id = os.getenv("AZURE_OPENAI_CLIENT_ID")
    client_secret = os.getenv("AZURE_OPENAI_CLIENT_SECRET")

    missing = [
        name
        for name, val in [
            ("AZURE_OPENAI_TENANT_ID", tenant_id),
            ("AZURE_OPENAI_CLIENT_ID", client_id),
            ("AZURE_OPENAI_CLIENT_SECRET", client_secret),
        ]
        if not val
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required Azure credentials in .env: {', '.join(missing)}"
        )

    # Values are guaranteed non-None after the check above
    credential = ClientSecretCredential(
        tenant_id=str(tenant_id),
        client_id=str(client_id),
        client_secret=str(client_secret),
    )
    return get_bearer_token_provider(credential, _AZURE_OPENAI_SCOPE)


def get_azure_model(
    deployment: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    use_advanced_model: bool = False,
    **kwargs,
) -> AzureChatOpenAI:
    """
    Create and return a LangChain AzureChatOpenAI model instance authenticated
    via an Azure Service Principal (client credentials).

    Args:
        deployment:          Azure deployment name. Defaults to AZURE_OPENAI_DEPLOYMENT env var.
        model:               Model name. Defaults to AZURE_OPENAI_MODEL (or AZURE_OPENAI_ADVANCED_MODEL
                             when use_advanced_model=True).
        temperature:         Sampling temperature (0 = deterministic, 1 = creative). Default 0.7.
        max_tokens:          Maximum tokens in the response. None means model default.
        use_advanced_model:  When True, uses AZURE_OPENAI_ADVANCED_MODEL instead of AZURE_OPENAI_MODEL.
        **kwargs:            Any additional keyword arguments forwarded to AzureChatOpenAI.

    Returns:
        A configured AzureChatOpenAI instance.

    Example:
        from model import get_azure_model

        llm = get_azure_model()
        response = llm.invoke("Hello, world!")
        print(response.content)

        # Use the advanced deployment:
        advanced_llm = get_azure_model(use_advanced_model=True)
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_VERSION", "2024-05-01-preview")

    if not endpoint:
        raise EnvironmentError(
            "AZURE_OPENAI_ENDPOINT is not set in the environment / .env file."
        )

    # Resolve deployment and model names
    if use_advanced_model:
        resolved_deployment = deployment or os.getenv(
            "AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"
        )
        resolved_model = model or os.getenv("AZURE_OPENAI_ADVANCED_MODEL", "gpt-4o-mini")
    else:
        resolved_deployment = deployment or os.getenv(
            "AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"
        )
        resolved_model = model or os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")

    token_provider = _build_token_provider()

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=resolved_deployment,
        api_version=api_version,
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        azure_ad_token_provider=token_provider,
        **kwargs,
    )



