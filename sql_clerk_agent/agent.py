from llm import get_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware, PIIMiddleware

db = SQLDatabase.from_uri(
"postgresql+psycopg2://postgres:root@localhost:5432/postgres"
)

model = get_model("gemini")
toolkit = SQLDatabaseToolkit(db=db,llm=model)
tools = toolkit.get_tools()

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="You are a SQL clerk agent. Use the tools to retrieve student details and return them.",
    middleware=[ModelRetryMiddleware(max_retries=3),
                ToolRetryMiddleware(max_retries=3)]
)