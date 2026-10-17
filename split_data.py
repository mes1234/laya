import argparse
import json
import random
from pathlib import Path

from machine_condition import load_records

BASE_DIR = Path(__file__).resolve().parent


def split_records(records: list[dict], ratio: float, seed: int | None = None) -> tuple[list[dict], list[dict]]:
    """Shuffle and split records; `ratio` is the share that goes to training."""
    if not 0 < ratio < 1:
        raise ValueError("ratio must be greater than 0 and less than 1.")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_count = round(len(shuffled) * ratio)
    return shuffled[:train_count], shuffled[train_count:]


def write_records(records: list[dict], path: Path) -> None:
    """One JSON object per line, the same format the playground writes."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split data.json into training and verification sets.")
    parser.add_argument("ratio", type=float, nargs="?", default=0.8,
                        help="share of records for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for a reproducible split")
    parser.add_argument("--input", type=Path, default=BASE_DIR / "data.json")
    args = parser.parse_args()

    records = load_records(args.input)
    if not records:
        raise SystemExit(f"{args.input.name} has no records.")
    try:
        train, verification = split_records(records, args.ratio, args.seed)
    except ValueError as error:
        raise SystemExit(str(error))

    write_records(train, args.input.with_name("data_train.json"))
    write_records(verification, args.input.with_name("data_verification.json"))
    print(f"Split {len(records)} records: {len(train)} train, {len(verification)} verification.")


if __name__ == "__main__":
    main()
