from pathlib import Path

import torch
from rl_agent_api import RLAgent

model_dir = Path(__file__).resolve().parent

device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

model = RLAgent(model_dir=model_dir, device=device)  # or "cpu"

print("Model loaded successfully.")

# Assume the user already told us their mood
mood = "tired and relaxed"

state = (
    f"The user says: 'I feel {mood}. "
    "I want to know what to do this evening.'"
)

questions = {
    "evening_activity": {
        "type": "choice",  # "choice" | "score" | "noul"
        "instructions": (
            "Based on the user's mood, choose the best evening activity "
            "from the given options."
        ),
        "criteria": ["gym", "sleep", "watch tv"]
    }
}
for i in range(300):
    for mood in ["tired and relaxed", "stressed and energetic", "bored and lazy"]:
        state = (
            f"The user says: 'I feel {mood}. "
            "I want to know what to do this evening.'"
        )
        print(model.system_one(state=state, questions=questions))