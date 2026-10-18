import random
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
    def __init__(self, rows, shuffle=True, seed=None):
        self.rows = rows
        self.shuffle = shuffle
        self.random = random.Random(seed)
        self.order = list(range(len(rows)))
        self.index = 0

    def has_next(self):
        return self.index < len(self.rows)

    def reset(self):
        self.index = 0
        if self.shuffle:
            self.random.shuffle(self.order)

    def fetch_batch(self, size):
        if not self.has_next():
            raise StopIteration
        indices = self.order[self.index:self.index + size]
        self.index += len(indices)
        return [
            {
                "state": self.rows[i]["story"],
                "label": CORRECT_TO_CRITERION[self.rows[i]["correct"]],
            }
            for i in indices
        ]


def encode_batch(model, examples):
    """Pad a list of {"state", "label"} dicts into one model batch and its target tensor."""
    q = model._to_internal(QUESTIONS[QUESTION_ID])
    items = []
    for example in examples:
        seq, markers = build_sequence(
            model.tok,
            example["state"],
            q,
            model.cfg["max_len"],
            model.cfg["head_max_len"],
        )
        items.append(
            {
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES["choice"],
                "target": [0.0] * len(markers),
                "label": CRITERIA.index(example["label"]),
                "episode": 0,
                "ep_step": 0,
                "ep_len": 1,
                "src": "train",
            }
        )

    batch = collate_items([items], model.tok.pad_token_id)
    target = torch.tensor(
        [CRITERIA.index(example["label"]) for example in examples],
        dtype=torch.long,
        device=model.device,
    )

    return {
        k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }, target


def forward_batch(model, batch):
    logits, _ = model.model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["marker_pos"],
        batch["marker_mask"],
        batch["qtype"],
    )
    return logits


def set_train_mode(model):
    model.model.train()
    # The frozen BERT encoder also stays free of dropout.
    model.model.encoder.eval()


def evaluate(model, rows, batch_size=8):
    """Loss, accuracy, per-class recall, confusion matrix and wrong stories."""
    model.model.eval()
    size = len(CRITERIA)
    confusion = torch.zeros(size, size, dtype=torch.long)
    wrong = []
    total_loss = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            examples = [
                {"state": row["story"], "label": CORRECT_TO_CRITERION[row["correct"]]}
                for row in chunk
            ]
            batch, target = encode_batch(model, examples)
            logits = forward_batch(model, batch)
            total_loss += F.cross_entropy(logits, target, reduction="sum").item()
            for example, actual, predicted in zip(
                examples, target.tolist(), logits.argmax(dim=-1).tolist()
            ):
                confusion[actual, predicted] += 1
                if predicted != actual:
                    wrong.append((example["state"], CRITERIA[actual], CRITERIA[predicted]))
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


def clone_state_dict(model):
    # Clone to CPU so tied/shared tensors don't break safetensors serialization.
    return {
        name: tensor.detach().cpu().clone().contiguous()
        for name, tensor in model.model.state_dict().items()
    }


def train_rl_agent(
    model,
    training_object,
    epochs=300,
    lr=1e-4,
    writer=None,
    validation_rows=None,
    confusion_every=10,
    batch_size=8,
    max_grad_norm=1.0,
):
    """Train and return (model, best_state_dict, best_info).

    The best epoch is the lowest validation loss, or the lowest training loss
    when there are no validation rows.
    """
    set_train_mode(model)
    # Keep the BERT encoder frozen: no gradient updates.
    for parameter in model.model.encoder.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (p for p in model.model.parameters() if p.requires_grad), lr=lr
    )

    trainable = [p for p in model.model.parameters() if p.requires_grad]
    best_loss = float("inf")
    best_state = None
    best_info = {}

    for epoch in range(epochs):
        total_loss = 0.0
        total_grad_norm = 0.0
        correct = 0
        count = 0
        batches = 0
        train_wrong = []

        training_object.reset()
        while training_object.has_next():
            examples = training_object.fetch_batch(batch_size)
            batch, target = encode_batch(model, examples)

            optimizer.zero_grad(set_to_none=True)

            logits = forward_batch(model, batch)

            loss = F.cross_entropy(logits, target)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()

            total_loss += loss.item() * len(examples)
            total_grad_norm += float(grad_norm)
            batches += 1
            for example, actual, predicted in zip(
                examples, target.tolist(), logits.argmax(dim=-1).tolist()
            ):
                if predicted == actual:
                    correct += 1
                else:
                    train_wrong.append((example["state"], example["label"], CRITERIA[predicted]))
            count += len(examples)

        mean_loss = total_loss / max(1, count)
        accuracy = correct / max(1, count)
        mean_grad_norm = total_grad_norm / max(1, batches)
        step = epoch + 1
        message = f"epoch={step} loss={mean_loss:.4f} accuracy={accuracy:.2%} grad_norm={mean_grad_norm:.3f}"
        selection_loss = mean_loss
        if writer is not None:
            writer.add_scalar("train/loss", mean_loss, step)
            writer.add_scalar("train/accuracy", accuracy, step)
            writer.add_scalar("train/grad_norm", mean_grad_norm, step)
            writer.add_text("train/misclassified", format_wrong(train_wrong), step)

        if validation_rows:
            result = evaluate(model, validation_rows, batch_size)
            selection_loss = result["loss"]
            message += f" val_loss={result['loss']:.4f} val_accuracy={result['accuracy']:.2%}"
            if writer is not None:
                writer.add_scalar("val/loss", result["loss"], step)
                writer.add_scalar("val/accuracy", result["accuracy"], step)
                for name, recall in result["recall"].items():
                    writer.add_scalar(f"val/recall/{name}", recall, step)
                writer.add_text("val/misclassified", format_wrong(result["wrong"]), step)
                if step % confusion_every == 0 or step == epochs:
                    writer.add_figure("val/confusion_matrix", confusion_figure(result["confusion"]), step)

        if selection_loss < best_loss:
            best_loss = selection_loss
            best_info = {"epoch": step, "loss": selection_loss}
            best_state = clone_state_dict(model)
            message += " *best*"
            if writer is not None:
                writer.add_scalar("best/epoch", step, step)

        if writer is not None:
            writer.flush()
        print(message)

    return model, best_state, best_info


model_dir = Path(__file__).resolve().parent
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

model = RLAgent(model_dir=model_dir, device=device, load_checkpoint=False)

training_rows = load_records(model_dir / "data_train.json")
if not training_rows:
    raise SystemExit("data.json has no records to train on.")

verification_path = model_dir / "data_verification.json"
verification_rows = load_records(verification_path) if verification_path.exists() else []
if not verification_rows:
    print("No data_verification.json records found; skipping validation metrics.")

training_object = MachineConditionTrainingObject(training_rows, seed=0)
writer = SummaryWriter(log_dir=str(model_dir / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")))
try:
    model, best_state, best_info = train_rl_agent(
        model,
        training_object,
        epochs=100,
        writer=writer,
        validation_rows=verification_rows,
        batch_size=8,
        max_grad_norm=1.0,
    )
finally:
    writer.close()

output_path = model_dir / "trained.safetensors"
save_file(
    best_state,
    str(output_path),
    metadata={"format": "pt", "encoder": model.cfg["encoder"]},
)
model.tok.save_pretrained(model_dir / "tokenizer")
metric = "val_loss" if verification_rows else "train_loss"
print(
    f"Saved best model (epoch {best_info['epoch']}, {metric}={best_info['loss']:.4f}) to {output_path}"
)
