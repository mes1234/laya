from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from machine_condition import CORRECT_TO_CRITERION, CRITERIA, QUESTION_ID, QUESTIONS, load_records
from rl_agent_api import RLAgent, build_sequence, collate_items, QTYPES


class MachineConditionTrainingObject:
    def __init__(self, rows):
        self.rows = rows
        self.index = 0

    def has_next(self):
        return self.index < len(self.rows)

    def reset(self):
        self.index = 0

    def fetch(self):
        if not self.has_next():
            raise StopIteration
        row = self.rows[self.index]
        self.index += 1
        return {
            "state": row["story"],
            "label": CORRECT_TO_CRITERION[row["correct"]],
        }


def encode_example(model, state, label):
    q = model._to_internal(QUESTIONS[QUESTION_ID])
    seq, markers = build_sequence(
        model.tok,
        state,
        q,
        model.cfg["max_len"],
        model.cfg["head_max_len"],
    )

    item = {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES["choice"],
        "target": [0.0] * len(markers),
        "label": CRITERIA.index(label),
        "episode": 0,
        "ep_step": 0,
        "ep_len": 1,
        "src": "train",
    }

    batch = collate_items([[item]], model.tok.pad_token_id)
    target = torch.tensor([CRITERIA.index(label)], dtype=torch.long, device=model.device)

    return {
        k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }, target


def train_rl_agent(model, training_object, epochs=300, lr=1e-4):
    model.model.train()
    # Keep the BERT encoder frozen: no gradient updates and no dropout.
    for parameter in model.model.encoder.parameters():
        parameter.requires_grad = False
    model.model.encoder.eval()
    optimizer = torch.optim.AdamW(
        (p for p in model.model.parameters() if p.requires_grad), lr=lr
    )

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0

        training_object.reset()
        while training_object.has_next():
            row = training_object.fetch()
            batch, target = encode_example(model, row["state"], row["label"])

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
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

model = RLAgent(model_dir=model_dir, device=device)

training_rows = load_records(model_dir / "data.json")
if not training_rows:
    raise SystemExit("data.json has no records to train on.")

training_object = MachineConditionTrainingObject(training_rows)
model = train_rl_agent(model, training_object, epochs=50)

# Clone to CPU so tied/shared tensors don't break safetensors serialization.
state_dict = {
    name: tensor.detach().cpu().clone().contiguous()
    for name, tensor in model.model.state_dict().items()
}
output_path = model_dir / "trained.safetensors"
save_file(state_dict, str(output_path), metadata={"format": "pt"})
print(f"Saved trained model to {output_path}")
