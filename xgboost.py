import random

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from dataclasses import dataclass


@dataclass
class Data:
    x: float
    y: float    
    z: float


@dataclass
class Prediction:
    intensity: float # 0-1
    width: float # 0-1
    length: float # 0-1

    def __str__(self) -> str:
        return (
            f"Prediction(intensity={self.intensity:.4f}, "
            f"width={self.width:.4f}, length={self.length:.4f})"
        )


@dataclass
class ModelInput:
    data: Data
    correct: Prediction

def generate(input: Prediction) -> ModelInput:
    # Example implementation, replace with actual logic
    return ModelInput(
        data=Data(
            x=input.intensity * input.width + random.random()/2,
            y=input.width **2 + random.random()/10,
            z=input.length / input.width + random.random()/10
        ),
        correct=input
    )


def generate_monte_carlo(
    n: int,
    intensity_range: tuple[float, float],
    width_range: tuple[float, float],
    length_range: tuple[float, float],
    seed: int | None = None,
) -> list[ModelInput]:
    """Create n ModelInput items by sampling Prediction values uniformly from the given (min, max) ranges."""
    for name, (lo, hi) in (
        ("intensity_range", intensity_range),
        ("width_range", width_range),
        ("length_range", length_range),
    ):
        if not 0 <= lo <= hi <= 1:
            raise ValueError(f"{name} must satisfy 0 <= min <= max <= 1, got {(lo, hi)}")
    if width_range[1] == 0:
        raise ValueError("width_range max must be > 0 (width is used as a divisor)")

    rng = random.Random(seed)
    items: list[ModelInput] = []
    while len(items) < n:
        width = rng.uniform(*width_range)
        if width == 0:  # avoid division by zero in generate()
            continue
        items.append(
            generate(
                Prediction(
                    intensity=rng.uniform(*intensity_range),
                    width=width,
                    length=rng.uniform(*length_range),
                )
            )
        )
    return items


def to_features(data: Data) -> list[float]:
    return [data.x, data.y, data.z]


def to_targets(p: Prediction) -> list[float]:
    return [p.intensity, p.width, p.length]


def train_model(items: list[ModelInput], **xgb_params) -> MultiOutputRegressor:
    """Fit a multi-output XGBoost regressor mapping Data -> (intensity, width, length)."""
    X = np.array([to_features(i.data) for i in items])
    y = np.array([to_targets(i.correct) for i in items])
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, **xgb_params}
    model = MultiOutputRegressor(XGBRegressor(**params))
    return model.fit(X, y)


def predict(model: MultiOutputRegressor, data: Data) -> Prediction:
    intensity, width, length = model.predict(np.array([to_features(data)]))[0]
    return Prediction(float(intensity), float(width), float(length))


def rms_error(true: Prediction, pred: Prediction) -> float:
    diff = np.array(to_targets(pred)) - np.array(to_targets(true))
    return float(np.sqrt((diff**2).mean()))


def run():
    items = generate_monte_carlo(
        n=20000,
        intensity_range=(0, 1),
        width_range=(0.1, 1),
        length_range=(0, 1),
        seed=42,
    )
    split = int(len(items) * 0.8)
    train, test = items[:split], items[split:]
    model = train_model(train)

    for item in test[:5]:
        predicted = predict(model, item.data)
        rms = rms_error(item.correct, predicted)
        print(f"true: {item.correct}")
        print(f"pred: {predicted}")
        print(f"RMS error: {rms:.5f}\n")

    X = np.array([to_features(i.data) for i in test])
    y = np.array([to_targets(i.correct) for i in test])
    err = model.predict(X) - y
    per_output = np.sqrt((err**2).mean(axis=0))
    print(
        "RMSE per output: "
        f"intensity={per_output[0]:.5f}, width={per_output[1]:.5f}, length={per_output[2]:.5f}"
    )
    print(f"RMSE overall: {np.sqrt((err**2).mean()):.5f}")


if __name__ == "__main__":
    run()
