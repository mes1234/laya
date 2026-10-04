from pathlib import Path

import torch
import torch.nn.functional as F

from rl_agent_api import RLAgent, build_sequence, collate_items, QTYPES


class ScamLegitTrainingObject:
    def __init__(self, rows):
        self.rows = rows
        self.index = 0

    def has_next(self):
        return self.index < len(self.rows)

    def fetch(self):
        if not self.has_next():
            raise StopIteration
        row = self.rows[self.index]
        self.index += 1
        return {
            "message": row["message"],
            "label": row["label"],  # "scam" or "legit"
        }


def encode_scam_example(model, message, label):
    question = {
        "type": "choice",
        "instructions": "Decide if this message is scam or legit.",
        "criteria": ["scam", "legit"],
    }

    q = model._to_internal(question)
    seq, markers = build_sequence(
        model.tok,
        message,
        q,
        model.cfg["max_len"],
        model.cfg["head_max_len"],
    )

    item = {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES["choice"],
        "target": [0.0] * len(markers),
        "label": 0,
        "episode": 0,
        "ep_step": 0,
        "ep_len": 1,
        "src": "train",
    }

    # This is the fix: batch must be a list of groups, each group a list of items
    batch = collate_items([[item]], model.tok.pad_token_id)
    target = torch.tensor([0 if label == "scam" else 1], dtype=torch.long, device=model.device)

    return {
        k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }, target


def train_rl_agent(model, training_object, epochs=3, lr=1e-4):
    model.model.train()
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0

        while training_object.has_next():
            row = training_object.fetch()
            batch, target = encode_scam_example(model, row["message"], row["label"])

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model.model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["marker_pos"],
                batch["marker_mask"],
                batch["qtype"],
            )

            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        print(f"epoch={epoch + 1} loss={total_loss / max(1, count):.4f}")

    return model


model_dir = Path(__file__).resolve().parent
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = RLAgent(model_dir=model_dir, device=device)

training_rows = [
    {"message": "URGENT! Your bank account is locked. Click here now to verify.", "label": "scam"},
    {"message": "Hi, I will meet you at the office at 3pm.", "label": "legit"},
    {"message": "We noticed suspicious activity on your card. Please call us immediately.", "label": "scam"},
    {"message": "Thanks for the meeting. I will send the document later today.", "label": "legit"},
]

training_object = ScamLegitTrainingObject(training_rows)
model = train_rl_agent(model, training_object, epochs=5)