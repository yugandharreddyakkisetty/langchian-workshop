from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware

from llm import get_model
from nongraph_clerk_agent.clerk.UserContext import UserContext
from nongraph_clerk_agent.clerk.tool import (
    get_student_attendance_information,
    get_student_backlogs_information,
    get_student_personal_information,
    get_students_by_total_percentage,
)


clerk = create_agent(
    model=get_model("bedrock"),           # disable_streaming=True by default → no outputConfig sent
    context_schema=UserContext,
    tools=[
        get_student_personal_information,
        get_student_attendance_information,
        get_student_backlogs_information,
        get_students_by_total_percentage,
    ],
    system_prompt="You are a student records assistant. Use the tools to retrieve student details and return them.",
    middleware=[ModelRetryMiddleware(max_retries=3), ToolRetryMiddleware(max_retries=3)],
)

