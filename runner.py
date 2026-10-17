from pathlib import Path

import torch
from safetensors.torch import load_file
from machine_condition import CORRECT_TO_CRITERION, CRITERIA, QUESTION_ID, QUESTIONS, load_records
from rl_agent_api import RLAgent

model_dir = Path(__file__).resolve().parent

device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

model = RLAgent(model_dir=model_dir, device=device, model="trained.safetensors")  # or "cpu"

print("Model loaded successfully.")

records = load_records(model_dir / "data_verification.json")

STORY_WIDTH = 40
header = (
    ["#", "Result", "Expected", "Predicted", "Confidence", "Act prob", "Tokens"]
    + [f"P({c})" for c in CRITERIA]
    + ["Story"]
)
rows = []
matches = 0
model_name = "-"
for index, record in enumerate(records, start=1):
    response = model.system_one(state=record["story"], questions=QUESTIONS)
    model_name = response["model"]
    answer = response["answers"][QUESTION_ID]
    predicted = answer["choice"]
    expected = CORRECT_TO_CRITERION[record["correct"]]
    matched = predicted == expected
    matches += matched
    story = record["story"]
    if len(story) > STORY_WIDTH:
        story = story[: STORY_WIDTH - 3] + "..."
    rows.append(
        [
            str(index),
            "OK" if matched else "FAIL",
            expected,
            predicted,
            f"{answer['confidence']:.2%}",
            f"{answer['rl_agent']['act_probability']:.2%}",
            str(response["usage"]["input_tokens"]),
        ]
        + [f"{answer['probabilities'].get(c, 0.0):.2%}" for c in CRITERIA]
        + [story]
    )


def print_table(header, rows):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    print(line)
    print(fmt.format(*header))
    print(line)
    for row in rows:
        print(fmt.format(*row))
    print(line)


if records:
    print(f"Model: {model_name}   Question: {QUESTION_ID}")
    print_table(header, rows)
    print(f"Matched {matches}/{len(records)} ({matches / len(records):.0%})")
else:
    print("data_verification.json has no records.")
