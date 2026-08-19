from langchain.messages import SystemMessage,AIMessage,HumanMessage,ToolMessage

from llm import get_model
from math_agent.utils.state import MessagesState
from math_agent.utils.tools import add,multiply,divide
from langgraph.graph import StateGraph,START,END
from typing import Literal

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}

model = get_model("bedrock").bind_tools(tools=tools)

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model.invoke(
                [
                    SystemMessage(
                        content="You are a student records assistant. Use the tools to retrieve student details and return them"
                    )
                ]
                + state["messages"]
            )
        ]
    }

# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

