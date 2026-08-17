from langgraph.graph import StateGraph
from graph_clerk_agent.utils.state import MessagesState, UserContext
from graph_clerk_agent.utils.nodes import llm_call, tool_node, should_continue
from langgraph.graph import START, END
# Build workflow
agent_builder = StateGraph(MessagesState,context_schema=UserContext)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
clerk = agent_builder.compile()