from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolRuntime
from langchain_core.tools import InjectedToolArg
from typing import Any, Annotated
from langchain.tools import tool

# from langchain_core.runnables import RunnableConfig  # Alternative: RunnableConfig approach

from graph_clerk_agent.utils.state import UserContext


@tool
def get_student_details(runtime: Annotated[ToolRuntime[UserContext], InjectedToolArg],
                        config:RunnableConfig) -> dict | None:
    """ Get student details by number """
    # Alternative using RunnableConfig (no InjectedToolArg needed, works with custom tool_node):
    # def get_student_details(config: RunnableConfig) -> dict | None:
    #     number = config["configurable"].get("number")
    thread_id = config.get("configurable", {}).get("thread_id", None)
    print("[INFO] get_student_details called with thread_id:", thread_id)
    students: dict[str, Any] = {
        "14001": {"name": "Yugandhar", "age": 22, "roll_number": "14001"},
        "14002": {"name": "Manoj", "age": 23, "roll_number": "14002"},
        "14003": {"name": "Nagesh", "age": 24, "roll_number": "14003"},
    }
    number = runtime.context.number
    return students.get(number, None)


@tool
def get_student_age_by_number(runtime: Annotated[ToolRuntime[UserContext], InjectedToolArg],
                              config: RunnableConfig) -> str:
    """ Get student age by number """
    # Alternative using RunnableConfig (no InjectedToolArg needed, works with custom tool_node):
    # def get_student_age_by_number(config: RunnableConfig) -> str:
    #     number = config["configurable"].get("number")
    students: dict[str, str] = {
        "14001": "22",
        "14002": "23",
        "14003": "24",
    }
    thread_id = config.get("configurable", {}).get("thread_id", None)
    print("[INFO] get_student_age_by_number called with thread_id:", thread_id)
    number = runtime.context.number
    age = students.get(number)
    if age:
        return f"Student #{number} age: {age}"
    else:
        return f"Student #{number}: N/A"