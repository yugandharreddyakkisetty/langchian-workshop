"""
Google Gemini model factory using LangChain.
============================================

Authentication uses a Google API key.

Required .env variable:
    GOOGLE_API_KEY     – your Google AI Studio API key

Optional .env variables:
    GEMINI_MODEL_ID    – Gemini model ID  (default: gemini-2.0-flash)

Usage
-----
    from llm import get_gemini_model

    llm = get_gemini_model()
    response = llm.invoke("What is LangChain?")
    print(response.content)

    # Use a specific model:
    llm = get_gemini_model(model="gemini-2.0-pro")
"""

import os
import warnings
from dotenv import load_dotenv

# Suppress noisy warnings from google-genai / gRPC / protobuf / pydantic
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"google.*")
warnings.filterwarnings("ignore", category=UserWarning,        module=r"langchain_google_genai.*")
warnings.filterwarnings("ignore", category=UserWarning,        module=r"google.*")
warnings.filterwarnings("ignore", message=r".*grpc.*",         category=RuntimeWarning)
warnings.filterwarnings("ignore", message=r".*Convert_system_message.*")
warnings.filterwarnings("ignore", message=r".*PydanticDeprecated.*")

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

_DEFAULT_MODEL_ID = "gemini-2.0-flash"


def get_gemini_model(
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs,
) -> ChatGoogleGenerativeAI:
    """
    Create and return a LangChain ChatGoogleGenerativeAI model instance.

    Args:
        model:       Gemini model ID. Defaults to GEMINI_MODEL_ID env var,
                     then to ``gemini-2.0-flash``.
        temperature: Sampling temperature (0 = deterministic, 1 = creative).
                     Default 0.7.
        max_tokens:  Maximum tokens in the response. None means model default.
        **kwargs:    Any additional keyword arguments forwarded to
                     ChatGoogleGenerativeAI.

    Returns:
        A configured ChatGoogleGenerativeAI instance.

    Raises:
        EnvironmentError: If GOOGLE_API_KEY is missing.

    Example:
        from llm import get_gemini_model

        llm = get_gemini_model()
        response = llm.invoke("Hello, Gemini!")
        print(response.content)

        # Use a specific model:
        llm = get_gemini_model(model="gemini-2.0-pro")
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set in the environment / .env file."
        )

    resolved_model = model or os.getenv("GEMINI_MODEL_ID", _DEFAULT_MODEL_ID)

    return ChatGoogleGenerativeAI(
        model=resolved_model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        google_api_key=api_key,
        **kwargs,
    )

