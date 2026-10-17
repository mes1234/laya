from pathlib import Path

import torch
from safetensors.torch import load_file
from machine_condition import CORRECT_TO_CRITERION, QUESTION_ID, QUESTIONS, load_records
from rl_agent_api import RLAgent

model_dir = Path(__file__).resolve().parent

device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

model = RLAgent(model_dir=model_dir, device=device, model="trained.safetensors")  # or "cpu"

print("Model loaded successfully.")

records = load_records(model_dir / "data.json")
matches = 0
for record in records:
    answer = model.system_one(state=record["story"], questions=QUESTIONS)
    predicted = answer["answers"][QUESTION_ID]["choice"]
    expected = CORRECT_TO_CRITERION[record["correct"]]
    matched = predicted == expected
    matches += matched
    print(f"{'OK  ' if matched else 'FAIL'} predicted={predicted!r} correct={expected!r} state={record['story']!r}\n\n")

if records:
    print(f"Matched {matches}/{len(records)} ({matches / len(records):.0%})")
else:
    print("data.json has no records.")
