"""
OpenAI model factory using LangChain.
Reads credentials from the .env file (OPENAI_API_KEY, OPENAI_MODEL, OPENAI_API_BASE).
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_openai_model(
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs,
) -> ChatOpenAI:
    """
    Create and return a LangChain ChatOpenAI model instance.

    Args:
        model:       Model name to use. Defaults to OPENAI_MODEL env var or 'gpt-4o-mini'.
        temperature: Sampling temperature (0 = deterministic, 1 = creative). Default 0.7.
        max_tokens:  Maximum tokens in the response. None means model default.
        **kwargs:    Any additional keyword arguments forwarded to ChatOpenAI.

    Returns:
        A configured ChatOpenAI instance.

    Example:
        from model import get_openai_model

        llm = get_openai_model()
        response = llm.invoke("Hello, world!")
        print(response.content)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment / .env file.")

    resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_API_BASE")  # optional; falls back to LangChain default

    init_kwargs = dict(
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        **kwargs,
    )
    if base_url:
        init_kwargs["base_url"] = base_url

    return ChatOpenAI(**init_kwargs)


