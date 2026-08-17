"""
AWS Bedrock model factory using LangChain (Converse API).
=========================================================

Authentication uses a Bedrock API key (bearer token) — no IAM credentials required.

Required .env variable:
    BEDROCK_API_KEY    – your AWS Bedrock API key

Optional .env variables:
    AWS_DEFAULT_REGION – AWS region          (default: us-east-1)
    BEDROCK_MODEL_ID   – Bedrock model ID    (default: anthropic.claude-3-5-sonnet-20241022-v2:0)

Usage
-----
    from llm import get_bedrock_model

    llm = get_bedrock_model()
    response = llm.invoke("What is LangChain?")
    print(response.content)
"""

import os
import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()

_DEFAULT_MODEL_ID = "openai.gpt-oss-safeguard-20b"
_DEFAULT_REGION   = "ap-south-1"


def get_bedrock_model(
    model_id: str | None = None,
    region: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    disable_streaming: bool = True,
    **kwargs,
) -> ChatBedrockConverse:
    """
    Create and return a LangChain ChatBedrockConverse model instance
    authenticated via a Bedrock API key (bearer token).

    Only BEDROCK_API_KEY is required — no AWS access/secret key needed.

    Args:
        model_id:     Bedrock model ID.  Defaults to BEDROCK_MODEL_ID env var,
                      then to ``anthropic.claude-3-5-sonnet-20241022-v2:0``.
        region:       AWS region.  Defaults to AWS_DEFAULT_REGION env var,
                      then to ``us-east-1``.
        temperature:  Sampling temperature (0 = deterministic, 1 = creative).
                      Default 0.7.
        max_tokens:   Maximum tokens in the response.  None means model default.
        **kwargs:     Any additional keyword arguments forwarded to
                      ChatBedrockConverse.

    Returns:
        A configured ChatBedrockConverse instance.

    Raises:
        EnvironmentError: If BEDROCK_API_KEY is missing.

    Example:
        from llm import get_bedrock_model

        llm = get_bedrock_model()
        response = llm.invoke("Hello, world!")
        print(response.content)

        # Use a specific model and region:
        llm = get_bedrock_model(
            model_id="amazon.nova-pro-v1:0",
            region="eu-central-1",
        )
    """
    # ── API key check ──────────────────────────────────────────────────────────
    api_key = os.getenv("BEDROCK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "BEDROCK_API_KEY is not set in the environment / .env file."
        )

    # ── Resolve model / region ─────────────────────────────────────────────────
    resolved_model_id = model_id or os.getenv("BEDROCK_MODEL_ID", _DEFAULT_MODEL_ID)
    resolved_region   = region   or os.getenv("AWS_DEFAULT_REGION", _DEFAULT_REGION)

    # ── Build boto3 client with bearer-token auth ──────────────────────────────
    # Setting AWS_BEARER_TOKEN_BEDROCK causes botocore to use the API key as a
    # bearer token instead of SigV4 (IAM) auth for all Bedrock service calls.
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=resolved_region,
    )

    return ChatBedrockConverse(
        model=resolved_model_id,
        region_name=resolved_region,
        client=bedrock_client,
        temperature=temperature,
        max_tokens=max_tokens,
        disable_streaming=disable_streaming,
        **kwargs,
    )
