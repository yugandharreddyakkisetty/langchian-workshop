from pydantic import BaseModel, Field

class Student(BaseModel):
    """Student Details"""
    name: str = Field(description="Student name")
    roll_number: int = Field(description="Student roll number")
    age: int = Field(description="Student age")




# ── JSON Schema dict for Student (no Pydantic model needed) ───────────────────
student_schema = {
    "title": "Student",
    "type": "object",
    "description": "Details of a student.",
    "properties": {
        "name":        {"type": "string",  "description": "Full name of the student"},
        "roll_number": {"type": "string",  "description": "Student roll number"},
        "age":         {"type": "integer", "description": "Age of the student"},
    },
    "required": ["name", "roll_number"],
}