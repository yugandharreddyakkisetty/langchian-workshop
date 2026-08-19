from langchain.messages import SystemMessage
from typing_extensions import Literal
from langgraph.prebuilt import ToolNode

from graph_clerk_agent.utils.state import MessagesState
from llm import get_model
from graph_clerk_agent.utils.tools import get_student_details, get_student_age_by_number
from langgraph.graph import END

tools = [get_student_details, get_student_age_by_number]

model = get_model("bedrock").bind_tools(tools=tools)

# Prebuilt ToolNode — automatically injects ToolRuntime from context_schema
tool_node = ToolNode(tools)

# Alternative: Custom tool_node using RunnableConfig (no ToolRuntime/InjectedToolArg needed):
# from langchain_core.runnables import RunnableConfig
# from langchain.messages import ToolMessage
# tools_by_name = {tool.name: tool for tool in tools}
# def tool_node(state: dict, config: RunnableConfig):
#     result = []
#     for tool_call in state["messages"][-1].tool_calls:
#         tool = tools_by_name[tool_call["name"]]
#         observation = tool.invoke(tool_call["args"], config)
#         result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
#     return {"messages": result}

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model.invoke(
                [SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")]
                + state["messages"]
            )
        ]
    }


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool_node"
    return END