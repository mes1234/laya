import json
from pathlib import Path

CRITERIA = ["Normal", "Noisy", "Gear Fail", "Bearing Fail"]
CORRECT_TO_CRITERION = {
    "Normal": "Normal",
    "Noisy": "Noisy",
    "Gear": "Gear Fail",
    "Bearing": "Bearing Fail",
}

QUESTION_ID = "machine_condition"
QUESTIONS = {
    QUESTION_ID: {
        "type": "choice",
        "instructions": (
            "Based on provided machine state decide what is condition of given machine."
        ),
        "criteria": CRITERIA,
    }
}


def load_records(path: Path) -> list[dict]:
    """Read records from data.json (JSON Lines, or a single JSON array)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]
