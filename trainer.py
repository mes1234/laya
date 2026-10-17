from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter

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


def set_train_mode(model):
    model.model.train()
    # The frozen BERT encoder also stays free of dropout.
    model.model.encoder.eval()


def evaluate(model, rows):
    """Loss, accuracy, per-class recall, confusion matrix and wrong stories."""
    model.model.eval()
    size = len(CRITERIA)
    confusion = torch.zeros(size, size, dtype=torch.long)
    wrong = []
    total_loss = 0.0
    with torch.no_grad():
        for row in rows:
            label = CORRECT_TO_CRITERION[row["correct"]]
            batch, target = encode_example(model, row["story"], label)
            logits, _ = model.model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["marker_pos"],
                batch["marker_mask"],
                batch["qtype"],
            )
            total_loss += F.cross_entropy(logits, target).item()
            predicted = int(logits.argmax(dim=-1).item())
            actual = int(target.item())
            confusion[actual, predicted] += 1
            if predicted != actual:
                wrong.append((row["story"], CRITERIA[actual], CRITERIA[predicted]))
    set_train_mode(model)

    count = max(1, len(rows))
    per_class_recall = {
        name: (confusion[i, i].item() / confusion[i].sum().item())
        for i, name in enumerate(CRITERIA)
        if confusion[i].sum().item() > 0
    }
    return {
        "loss": total_loss / count,
        "accuracy": confusion.diag().sum().item() / count,
        "recall": per_class_recall,
        "confusion": confusion,
        "wrong": wrong,
    }


def confusion_figure(confusion):
    figure = Figure(figsize=(5, 4))
    ax = figure.subplots()
    ax.imshow(confusion.numpy(), cmap="Blues")
    ax.set_xticks(range(len(CRITERIA)), CRITERIA, rotation=45, ha="right")
    ax.set_yticks(range(len(CRITERIA)), CRITERIA)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(CRITERIA)):
        for j in range(len(CRITERIA)):
            ax.text(j, i, int(confusion[i, j]), ha="center", va="center")
    figure.tight_layout()
    return figure


def format_wrong(wrong):
    if not wrong:
        return "none"
    return "\n\n".join(
        f"actual: {actual}  predicted: {predicted}  story: {story}"
        for story, actual, predicted in wrong
    )


def train_rl_agent(
    model,
    training_object,
    epochs=300,
    lr=1e-4,
    writer=None,
    validation_rows=None,
    confusion_every=10,
):
    set_train_mode(model)
    # Keep the BERT encoder frozen: no gradient updates.
    for parameter in model.model.encoder.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (p for p in model.model.parameters() if p.requires_grad), lr=lr
    )

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        count = 0
        train_wrong = []

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
            predicted = int(logits.argmax(dim=-1).item())
            if predicted == int(target.item()):
                correct += 1
            else:
                train_wrong.append((row["state"], row["label"], CRITERIA[predicted]))
            count += 1

        mean_loss = total_loss / max(1, count)
        accuracy = correct / max(1, count)
        step = epoch + 1
        message = f"epoch={step} loss={mean_loss:.4f} accuracy={accuracy:.2%}"
        if writer is not None:
            writer.add_scalar("train/loss", mean_loss, step)
            writer.add_scalar("train/accuracy", accuracy, step)
            writer.add_text("train/misclassified", format_wrong(train_wrong), step)

        if validation_rows:
            result = evaluate(model, validation_rows)
            message += f" val_loss={result['loss']:.4f} val_accuracy={result['accuracy']:.2%}"
            if writer is not None:
                writer.add_scalar("val/loss", result["loss"], step)
                writer.add_scalar("val/accuracy", result["accuracy"], step)
                for name, recall in result["recall"].items():
                    writer.add_scalar(f"val/recall/{name}", recall, step)
                writer.add_text("val/misclassified", format_wrong(result["wrong"]), step)
                if step % confusion_every == 0 or step == epochs:
                    writer.add_figure("val/confusion_matrix", confusion_figure(result["confusion"]), step)

        if writer is not None:
            writer.flush()
        print(message)

    return model


model_dir = Path(__file__).resolve().parent
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

model = RLAgent(model_dir=model_dir, device=device)

training_rows = load_records(model_dir / "data_train.json")
if not training_rows:
    raise SystemExit("data.json has no records to train on.")

verification_path = model_dir / "data_verification.json"
verification_rows = load_records(verification_path) if verification_path.exists() else []
if not verification_rows:
    print("No data_verification.json records found; skipping validation metrics.")

training_object = MachineConditionTrainingObject(training_rows)
writer = SummaryWriter(log_dir=str(model_dir / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")))
try:
    model = train_rl_agent(model, training_object, epochs=100, writer=writer, validation_rows=verification_rows)
finally:
    writer.close()

# Clone to CPU so tied/shared tensors don't break safetensors serialization.
state_dict = {
    name: tensor.detach().cpu().clone().contiguous()
    for name, tensor in model.model.state_dict().items()
}
output_path = model_dir / "trained.safetensors"
save_file(state_dict, str(output_path), metadata={"format": "pt"})
print(f"Saved trained model to {output_path}")
