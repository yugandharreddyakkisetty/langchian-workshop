import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain.tools import tool


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"


def _get_data_file_path(file_name: str) -> Path:
    """Find a student JSON file in the data folder."""
    file_path = DATA_DIR / file_name
    if file_path.exists():
        return file_path

    raise FileNotFoundError(f"Could not find {file_name}. Searched: {file_path}")


@lru_cache(maxsize=None)
def _load_json_file(file_name: str) -> dict[str, dict[str, Any]]:
    """Load a student JSON file from the data folder."""
    file_path = _get_data_file_path(file_name)
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_roll_number(roll_number: str) -> str:
    """Normalize roll number or hall ticket number for lookups."""
    return roll_number.strip().upper()


def _get_student_record(file_name: str, roll_number: str) -> dict[str, Any]:
    """Return one student record from a keyed JSON file."""
    normalized_roll_number = _normalize_roll_number(roll_number)

    if not normalized_roll_number:
        return {
            "roll_number": normalized_roll_number,
            "found": False,
            "message": "Roll number or hall ticket number is required.",
        }

    data = _load_json_file(file_name)
    record = data.get(normalized_roll_number)

    if record is None:
        return {
            "roll_number": normalized_roll_number,
            "found": False,
            "message": f"No student record found for roll number {normalized_roll_number}.",
        }

    return {
        "roll_number": normalized_roll_number,
        "found": True,
        "details": record,
    }


def _to_float(value: Any) -> float | None:
    """Convert a JSON value to float when possible."""
    if value is None:
        return None

    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _is_less_than_operator(operator: str) -> bool:
    """Return True when the operator means less than."""
    normalized_operator = operator.strip().lower()
    return normalized_operator in {"less", "less_than", "lt", "below", "under", "<"}


def _is_more_than_operator(operator: str) -> bool:
    """Return True when the operator means more than."""
    normalized_operator = operator.strip().lower()
    return normalized_operator in {"more", "more_than", "greater", "greater_than", "gt", "above", "over", ">"}


@tool
def get_student_personal_information(roll_number: str) -> dict[str, Any]:
    """Get a student's personal information by roll number or hall ticket number."""
    return _get_student_record("PersonalInfo.json", roll_number)


@tool
def get_student_backlogs_information(roll_number: str) -> dict[str, Any]:
    """Get a student's backlog, credits, CGPA, and percentage information by roll number or hall ticket number."""
    return _get_student_record("Backlogs.json", roll_number)


@tool
def get_student_attendance_information(roll_number: str) -> dict[str, Any]:
    """Get a student's subject-wise attendance, total attendance, and attendance percentage by roll number or hall ticket number."""
    return _get_student_record("Attendance.json", roll_number)


@tool
def get_students_by_total_percentage(
    percentage: float,
    operator: str,
    percentage_type: str = "academic",
) -> dict[str, Any]:
    """List students whose total percentage is less than or more than a number.

    Args:
        percentage: Percentage threshold to compare against.
        operator: Comparison operator. Use less, less_than, below, <, more, more_than, greater_than, above, or >.
        percentage_type: Use academic for Backlogs.json Percentage or attendance for Attendance.json PERC.
    """
    normalized_percentage_type = percentage_type.strip().lower()

    if normalized_percentage_type in {"attendance", "att"}:
        file_name = "Attendance.json"
        percentage_field = "PERC"
    elif normalized_percentage_type in {"academic", "academics", "total", "backlogs"}:
        file_name = "Backlogs.json"
        percentage_field = "Percentage"
    else:
        return {
            "found": False,
            "students": [],
            "message": "percentage_type must be either academic or attendance.",
        }

    if _is_less_than_operator(operator):
        comparison_name = "less_than"
        matches = lambda student_percentage: student_percentage < percentage
    elif _is_more_than_operator(operator):
        comparison_name = "more_than"
        matches = lambda student_percentage: student_percentage > percentage
    else:
        return {
            "found": False,
            "students": [],
            "message": "operator must be less_than or more_than.",
        }

    data = _load_json_file(file_name)
    students: list[dict[str, Any]] = []

    for roll_number, details in data.items():
        student_percentage = _to_float(details.get(percentage_field))
        if student_percentage is None or not matches(student_percentage):
            continue

        students.append(
            {
                "roll_number": roll_number,
                "name": details.get("Name", ""),
                "percentage": student_percentage,
                "details": details,
            }
        )

    students.sort(key=lambda student: student["percentage"])

    return {
        "found": bool(students),
        "percentage_type": normalized_percentage_type,
        "comparison": comparison_name,
        "threshold": percentage,
        "count": len(students),
        "students": students,
    }


