"""
Model Factory
=============
Single entry-point to get any supported LLM by passing a provider name.

Supported providers
-------------------
    "openai"   → ChatOpenAI              (standard OpenAI API)
    "azure"    → AzureChatOpenAI         (Azure OpenAI via Service Principal)
    "bedrock"  → ChatBedrockConverse      (AWS Bedrock Converse API)
    "gemini"   → ChatGoogleGenerativeAI  (Google Gemini via Google AI Studio)

Usage
-----
    from llm import get_model

    llm = get_model("openai")
    llm = get_model("azure")
    llm = get_model("bedrock")
    llm = get_model("gemini")
    llm = get_model("gemini", model="gemini-2.0-pro")

    response = llm.invoke("What is LangChain?")
    print(response.content)
"""

from enum import Enum
from typing import Callable
from langchain_core.language_models.chat_models import BaseChatModel

from .openai_model import get_openai_model
from .azure_model import get_azure_model
from .bedrock_model import get_bedrock_model
from .gemini_model import get_gemini_model


class ModelProvider(str, Enum):
    """Enum of all supported LLM providers."""
    OPENAI   = "openai"
    AZURE    = "azure"
    BEDROCK  = "bedrock"
    GEMINI   = "gemini"


# ── Registry ──────────────────────────────────────────────────────────────────
_REGISTRY: dict[ModelProvider, Callable] = {
    ModelProvider.OPENAI:   get_openai_model,
    ModelProvider.AZURE:    get_azure_model,
    ModelProvider.BEDROCK:  get_bedrock_model,
    ModelProvider.GEMINI:   get_gemini_model,
}


def get_model(provider: str | None = None, **kwargs) -> BaseChatModel:
    """
    Factory function — create and return a LangChain chat model for the given provider.

    Args:
        provider (str): Which LLM backend to use. Case-insensitive. One of:
                          - "openai"   → ChatOpenAI (standard OpenAI API)
                          - "azure"    → AzureChatOpenAI (Azure, Service Principal auth)
                          - "bedrock"  → ChatBedrockConverse (AWS Bedrock Converse API)
                          - "gemini"   → ChatGoogleGenerativeAI (Google Gemini)
        **kwargs:       Extra keyword arguments forwarded to the underlying factory.
                        Common options:
                          model          (str)   – override the model/deployment name
                          temperature    (float) – 0.0 (deterministic) … 1.0 (creative)
                          max_tokens     (int)   – max tokens in the response
                          use_advanced_model (bool) – Azure only
                          model_id       (str)   – Bedrock only: Bedrock model ID
                          region         (str)   – Bedrock only: AWS region

    Returns:
        BaseChatModel: A ready-to-use LangChain chat model instance.

    Raises:
        ValueError: If an unsupported provider string is passed.

    Examples:
        from llm import get_model

        llm = get_model("openai")
        llm = get_model("azure", use_advanced_model=True)
        llm = get_model("bedrock", model_id="amazon.nova-pro-v1:0")
        llm = get_model("gemini")
        llm = get_model("gemini", model="gemini-2.0-pro")
    """
    try:
        if provider is None:
            provider = "openai"
        key = ModelProvider(provider.strip().lower())
    except ValueError:
        supported = ", ".join(f'"{p.value}"' for p in ModelProvider)
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            f"Choose one of: {supported}."
        )

    factory = _REGISTRY[key]
    return factory(**kwargs)
