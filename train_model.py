"""Train the Fashion-MNIST ANN using the notebook's best Optuna parameters."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "fashion-mnist_train.csv"
CHECKPOINT_PATH = ROOT / "model.pth"

# Best parameters recorded by the Optuna notebook (trial 1).
PARAMS = {
    "num_hidden_layers": 4,
    "neurons_per_layer": 88,
    "epochs": 20,
    "learning_rate": 0.000300318300747871,
    "dropout_rate": 0.2,
    "batch_size": 128,
    "weight_decay": 0.0004539602814543209,
}


class MyNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_hidden_layers: int,
                 neurons_per_layer: int, dropout_rate: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(input_dim, neurons_per_layer),
                nn.BatchNorm1d(neurons_per_layer),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            input_dim = neurons_per_layer
        layers.append(nn.Linear(neurons_per_layer, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Training CSV not found: {CSV_PATH}")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Reading {CSV_PATH.name} on {device}…", flush=True)

    frame = pd.read_csv(CSV_PATH)
    if frame.shape != (60000, 785):
        raise ValueError(f"Expected 60,000 rows × 785 columns, got {frame.shape}")
    features = frame.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0
    labels = frame.iloc[:, 0].to_numpy(dtype=np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_loader = DataLoader(
        train_data, batch_size=PARAMS["batch_size"], shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    model = MyNN(
        input_dim=784,
        output_dim=10,
        num_hidden_layers=PARAMS["num_hidden_layers"],
        neurons_per_layer=PARAMS["neurons_per_layer"],
        dropout_rate=PARAMS["dropout_rate"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(
        model.parameters(), lr=PARAMS["learning_rate"],
        weight_decay=PARAMS["weight_decay"],
    )

    started = time.monotonic()
    for epoch in range(PARAMS["epochs"]):
        model.train()
        loss_sum = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(
            f"Epoch {epoch + 1:02d}/{PARAMS['epochs']}: "
            f"loss={loss_sum / len(train_loader):.4f} "
            f"elapsed={time.monotonic() - started:.0f}s",
            flush=True,
        )

    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for batch_x, batch_y in test_loader:
            predictions = model(batch_x.to(device)).argmax(dim=1).cpu()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = 100.0 * correct / total
    torch.save(model.cpu().state_dict(), CHECKPOINT_PATH)
    print(f"Held-out accuracy: {accuracy:.2f}% ({correct}/{total})", flush=True)
    print(f"Saved checkpoint: {CHECKPOINT_PATH} ({CHECKPOINT_PATH.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
