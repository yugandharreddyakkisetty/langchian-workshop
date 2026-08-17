"""
llm package
=============
Provides factory functions for creating LangChain LLM instances.

Usage
-----
    from llm import get_openai_model, get_azure_model, get_bedrock_model, get_gemini_model, get_model

    # Standard OpenAI
    llm = get_openai_model()

    # Azure OpenAI (Service Principal auth)
    llm = get_azure_model()

    # Azure OpenAI – advanced deployment
    llm = get_azure_model(use_advanced_model=True)

    # AWS Bedrock (Converse API)
    llm = get_bedrock_model()
    llm = get_bedrock_model(model_id="amazon.nova-pro-v1:0", region="eu-central-1")

    # Via factory
    llm = get_model("openai")
    llm = get_model("bedrock")
    llm = get_model("gemini")
"""

from .openai_model import get_openai_model
from .azure_model import get_azure_model
from .bedrock_model import get_bedrock_model
from .gemini_model import get_gemini_model
from .factory import get_model, ModelProvider

__all__ = [
    "get_openai_model",
    "get_azure_model",
    "get_bedrock_model",
    "get_gemini_model",
    "get_model",
    "ModelProvider",
]
