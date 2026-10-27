import ast
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATA_FILE = Path(__file__).with_name("attendance_data.json")
API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct:fastest")
MAX_TOOL_STEPS = 5

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_attendance",
            "description": "Read attendance and calculated percentages, optionally for one student.",
            "parameters": {
                "type": "object",
                "properties": {
                    "roll_no": {"type": "string", "description": "Student roll number; omit for all students."},
                    "subject": {"type": "string", "description": "Subject name; omit for all subjects."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_attendance",
            "description": "Mark or change a student's attendance for a subject and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "roll_no": {"type": "string"},
                    "subject": {"type": "string"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "status": {"type": "string", "enum": ["present", "absent"]},
                },
                "required": ["roll_no", "subject", "date", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_attendance",
            "description": "Remove a student's attendance record for a subject and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "roll_no": {"type": "string"},
                    "subject": {"type": "string"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                },
                "required": ["roll_no", "subject", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Calculate a basic arithmetic expression using +, -, *, and /.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


def load_database():
    with DATA_FILE.open("r", encoding="utf-8") as data_file:
        return json.load(data_file)


def save_database(database):
    with DATA_FILE.open("w", encoding="utf-8") as data_file:
        json.dump(database, data_file, indent=2, ensure_ascii=False)
        data_file.write("\n")


def find_student(database, roll_no):
    for student in database.get("students", []):
        if str(student.get("roll_no", "")).strip() == str(roll_no).strip():
            return student
    raise ValueError(f"Student with roll number '{roll_no}' was not found.")


def attendance_report(student, subject_filter=None):
    subjects = student.get("subjects", {})
    if subject_filter:
        matching_subjects = [name for name in subjects if name.lower() == subject_filter.lower()]
        if not matching_subjects:
            raise ValueError(f"Subject '{subject_filter}' was not found for {student['name']}.")
        subjects = {name: subjects[name] for name in matching_subjects}

    report = {
        "roll_no": student["roll_no"],
        "name": student["name"],
        "subjects": {},
    }
    for subject, records in subjects.items():
        days = records.get("days", [])
        attended = sum(day["status"] == "present" for day in days)
        total = len(days)
        report["subjects"][subject] = {
            "days": days,
            "classes_attended": attended,
            "total_classes": total,
            "attendance_percentage": round(attended * 100 / total, 2) if total else 0.0,
        }
    return report


def get_attendance(roll_no=None, subject=None):
    database = load_database()
    students = database.get("students", [])
    if roll_no is not None:
        students = [find_student(database, roll_no)]
    return [attendance_report(student, subject) for student in students]


def mark_attendance(roll_no, subject, date, status):
    if status not in {"present", "absent"}:
        raise ValueError("Status must be 'present' or 'absent'.")
    database = load_database()
    student = find_student(database, roll_no)
    subject_name = next((name for name in student["subjects"] if name.lower() == subject.lower()), None)
    if subject_name is None:
        raise ValueError(f"Subject '{subject}' was not found for {student['name']}.")

    days = student["subjects"][subject_name].setdefault("days", [])
    existing = next((day for day in days if day["date"] == date), None)
    if existing:
        existing["status"] = status
        action = "updated"
    else:
        days.append({"date": date, "status": status})
        days.sort(key=lambda day: day["date"])
        action = "marked"
    save_database(database)
    return {"result": action, "student": student["name"], "subject": subject_name, "date": date, "status": status}


def remove_attendance(roll_no, subject, date):
    database = load_database()
    student = find_student(database, roll_no)
    subject_name = next((name for name in student["subjects"] if name.lower() == subject.lower()), None)
    if subject_name is None:
        raise ValueError(f"Subject '{subject}' was not found for {student['name']}.")

    days = student["subjects"][subject_name].setdefault("days", [])
    remaining = [day for day in days if day["date"] != date]
    if len(remaining) == len(days):
        raise ValueError(f"No attendance record exists for {date} in {subject_name}.")
    student["subjects"][subject_name]["days"] = remaining
    save_database(database)
    return {"result": "removed", "student": student["name"], "subject": subject_name, "date": date}


def _calculate_node(node):
    operators = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.UAdd: lambda value: value,
        ast.USub: lambda value: -value,
    }
    if isinstance(node, ast.Expression):
        return _calculate_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in operators:
        return operators[type(node.op)](_calculate_node(node.left), _calculate_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
        return operators[type(node.op)](_calculate_node(node.operand))
    raise ValueError("Only numeric constants and +, -, *, / operations are allowed.")


def calculate_math(expression):
    result = _calculate_node(ast.parse(expression, mode="eval"))
    if not math.isfinite(result):
        raise ValueError("The calculation did not produce a finite number.")
    return {"expression": expression, "result": round(result, 2)}


TOOL_REGISTRY = {
    "get_attendance": get_attendance,
    "mark_attendance": mark_attendance,
    "remove_attendance": remove_attendance,
    "calculate_math": calculate_math,
}


def call_huggingface(messages):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Set it to a Hugging Face access token before asking a query.")
    body = json.dumps(
        {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": TOOLS_SCHEMA,
            "tool_choice": "auto",
            "temperature": 0,
        }
    ).encode("utf-8")
    request = Request(
        API_URL,
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))["choices"][0]["message"]
    except HTTPError as error:
        raise RuntimeError(f"Hugging Face API returned HTTP {error.code}.") from error
    except URLError as error:
        raise RuntimeError(f"Could not connect to Hugging Face: {error.reason}") from error


def ask_ai(user_query):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an attendance-management assistant. Use get_attendance for attendance facts, "
                "calculate_math for arithmetic, and the mark/remove tools for requested data changes. "
                "Do not claim a change succeeded unless its tool succeeded. The application executes "
                "tool calls and saves successful changes. Finish with a concise JSON object containing "
                "an 'answer' field."
            ),
        },
        {"role": "user", "content": user_query},
    ]
    tool_calls_used = []

    try:
        for _ in range(MAX_TOOL_STEPS):
            response_message = call_huggingface(messages)
            tool_calls = response_message.get("tool_calls") or []
            if not tool_calls:
                content = response_message.get("content") or ""
                try:
                    result = json.loads(content)
                    if not isinstance(result, dict):
                        result = {"answer": result}
                except (json.JSONDecodeError, TypeError):
                    result = {"answer": content}
                result["tools_used"] = tool_calls_used
                return result

            messages.append(response_message)
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = function.get("name", "")
                arguments = {}
                try:
                    arguments = json.loads(function.get("arguments", "{}"))
                    if name not in TOOL_REGISTRY:
                        raise ValueError(f"Unknown tool '{name}'.")
                    tool_result = TOOL_REGISTRY[name](**arguments)
                except Exception as error:
                    tool_result = {"error": str(error)}
                tool_calls_used.append({"name": name, "arguments": arguments, "result": tool_result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        return {
            "error": "The model exceeded the tool-call limit.",
            "tools_used": tool_calls_used,
        }
    except (RuntimeError, KeyError, IndexError, json.JSONDecodeError) as error:
        return {"error": str(error), "tools_used": tool_calls_used}



def handle_local_query(query):
    """Handle basic attendance queries without calling the AI API."""
    normalized = query.lower().replace("attendence", "attendance")

    if "attendance" not in normalized:
        return None

    words = normalized.replace("?", " ").replace(",", " ").split()
    roll_numbers = [
        word for word in words
        if word.isdigit()
    ]

    # If a roll number is supplied, show that student's attendance.
    if roll_numbers:
        return {
            "answer": "Student attendance report",
            "data": get_attendance(roll_no=roll_numbers[0]),
            "mode": "local"
        }

    # If the query asks for all students, show everyone.
    if any(word in words for word in ("all", "everyone", "students")):
        return {
            "answer": "Attendance report for all students",
            "data": get_attendance(),
            "mode": "local"
        }

    return None


if __name__ == "__main__":
    print("Attendance Assistant (Local attendance mode available).")
    print("Enter 'exit' to quit.")

    while True:
        try:
            query = input("\nQuery > ").strip()

            if query.lower() in {"exit", "quit", "q"}:
                break

            if not query:
                continue

            try:
                # First handle attendance directly from the JSON file.
                result = handle_local_query(query)

                if result is not None:
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                else:
                    # Use AI only for queries that need the language model.
                    print(json.dumps(
                        ask_ai(query),
                        indent=2,
                        ensure_ascii=False
                    ))

            except (ValueError, KeyError, FileNotFoundError,
                    json.JSONDecodeError) as error:
                print(json.dumps({"error": str(error)}, indent=2))

        except (KeyboardInterrupt, EOFError):
            print("\nProgram closed.")
            break
