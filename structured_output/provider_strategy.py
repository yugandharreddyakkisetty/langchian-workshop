from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolRuntime
from typing import Any
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware


from llm import get_model
from langchain.tools import tool

from structured_output.Student import Student, student_schema
from structured_output.UserContext import UserContext



@tool
def get_student_details(runtime : ToolRuntime[UserContext]) -> dict|None:
    """ Get student details by number """
    students: dict[str,Any] = {
        "14001": {
            "name": "Yugandhar",
            "age": 22,
            "roll_number": "14001"
        },
        "14002": {
            "name": "Manoj",
            "age": 23,
            "roll_number": "14002"
        },
        "14003":  {
            "name": "Nagesh",
            "age": 24,
            "roll_number": "14003"

        }
    }
    number = runtime.context.number
    student = students.get(number, None)
    return student



@tool
def get_student_age_by_number(runtime : ToolRuntime[UserContext]) -> str:
    """ Get student age by number """
    students: dict[str,str] = {
        "14001": "22",
        "14002": "23",
        "14003": "24"
    }
    number = runtime.context.number
    age = students.get(number)
    if age:
        return f"Student #{number} age: {age}"
    else:
        return f"Student #{number}: N/A"


clerk = create_agent(
    model=get_model("bedrock"),           # disable_streaming=True by default → no outputConfig sent
    context_schema=UserContext,
    tools=[get_student_details, get_student_age_by_number],
    response_format=ProviderStrategy(Student),  # Pydantic class → tool-call based structured output
    system_prompt="You are a student records assistant. Use the tools to retrieve student details and return them.",
    middleware=[ModelRetryMiddleware(max_retries=3), ToolRetryMiddleware(max_retries=3)],
)

response = clerk.invoke(
    {"messages": [HumanMessage(content="Get student details")]},
    context=UserContext(number="14002"),
)

student:Student = response["structured_response"]
print(student.name)
print(student.age)
print(student.roll_number)

print(response)
