from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int,runtime:ToolRuntime) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    print("Context from runtime: user_id", runtime.context.user_id)
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b