from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolRuntime
from dataclasses import dataclass

from llm import get_model

# ── JSON Schema dict (no Pydantic model needed) ───────────────────────────────
contact_info_schema = {
    "title": "ContactInfo",
    "type": "object",
    "description": "Contact information for a person.",
    "properties": {
        "name":  {"type": "string", "description": "The name of the person"},
        "email": {"type": "string", "description": "The email address of the person"},
        "phone": {"type": "string", "description": "The phone number of the person"},
    },
    "required": ["name", "email", "phone"],
}

# ── Runtime context ───────────────────────────────────────────────────────────
@dataclass
class StudentContext:
    student_id: str


# ── Tools ─────────────────────────────────────────────────────────────────────
STUDENT_DB: dict[str, dict] = {
    "S001": {"name": "Alice Johnson", "email": "alice@university.edu", "phone": "555-1001"},
    "S002": {"name": "Bob Smith",     "email": "bob@university.edu",   "phone": "555-1002"},
    "S003": {"name": "Carol White",   "email": "carol@university.edu", "phone": "555-1003"},
}


@tool
def get_student_contact(runtime: ToolRuntime[StudentContext]) -> str:
    """Get the contact information for the current student."""
    student_id = runtime.context.student_id
    print(f"[INFO] Fetching contact info for student_id: {student_id}")
    student = STUDENT_DB.get(student_id)
    if not student:
        return f"No student found with ID: {student_id}"
    return (
        f"Name: {student['name']}, "
        f"Email: {student['email']}, "
        f"Phone: {student['phone']}"
    )


@tool
def list_all_students(runtime: ToolRuntime[StudentContext]) -> str:
    """List all available student IDs."""
    ids = ", ".join(STUDENT_DB.keys())
    return f"Available student IDs: {ids}"


# ── Agent ─────────────────────────────────────────────────────────────────────
agent = create_agent(
    model=get_model("bedrock"),
    tools=[get_student_contact, list_all_students],
    context_schema=StudentContext,
    # Use ToolStrategy with a raw JSON schema dict — no Pydantic model required
    response_format=ToolStrategy(schema=contact_info_schema),
    system_prompt="You are a student records assistant. Retrieve and return student contact information.",
)

# ── Invoke ────────────────────────────────────────────────────────────────────
response = agent.invoke(
    {"messages": [HumanMessage(content="Get the contact details for this student.")]},
    context=StudentContext(student_id="S002"),
)

structured = response.get("structured_response")
if structured:
    print("\n=== STRUCTURED RESPONSE (ContactInfo) ===")
    print(f"Name:  {structured['name']}")
    print(f"Email: {structured['email']}")
    print(f"Phone: {structured['phone']}")
else:
    print("\n=== RAW RESPONSE ===")
    print(response["messages"][-1].content)

