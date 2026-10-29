# Threadline · Fashion Intelligence Studio

A responsive dashboard and inference API for the Fashion-MNIST PyTorch ANN in `Copy of ann-fashion-mnist-pytorch-gpu-optimized-optuna.ipynb`.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000. The dashboard can be explored without a checkpoint; live classification becomes available after adding the trained weights described below.

## Train and add the checkpoint

The generated `model.pth` was trained locally from the CSV and reached **88.77% held-out accuracy** on the notebook's 80/20 split. To reproduce it, run `python train_model.py` from this folder.

The notebook records a best validation accuracy of **89.08%**, with 4 hidden layers of 88 neurons and dropout 0.2. The notebook does **not** save trained weights, so an existing `study.best_params` result alone cannot create `model.pth`. Train the final model once, then save its state dict.

Add the following after the notebook's `MyNN` class and after `X_train`, `X_test`, `y_train`, `y_test`, `train_dataset`, and `test_dataset` have been prepared. It uses the best trial's settings, fixes the notebook's optimizer selection so the selected RMSprop optimizer is actually used, evaluates on the held-out split, and writes the checkpoint into this project folder:

```python
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

params = study.best_params
model = MyNN(
    input_dim=784,
    output_dim=10,
    num_hidden_layers=params["num_hidden_layers"],
    neurons_per_layer=params["neurons_per_layer"],
    dropout_rate=params["dropout_rate"],
).to(device)

train_loader = DataLoader(
    train_dataset, batch_size=params["batch_size"], shuffle=True,
    pin_memory=(device.type == "cuda"),
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(
    model.parameters(), lr=params["learning_rate"],
    weight_decay=params["weight_decay"],
)

for epoch in range(params["epochs"]):
    model.train()
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()

model.eval()
correct = total = 0
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
with torch.inference_mode():
    for features, labels in test_loader:
        logits = model(features.to(device))
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        total += labels.size(0)
print(f"Held-out accuracy: {100 * correct / total:.2f}%")

checkpoint_path = Path(r"C:\Users\acer\Downloads\pytorch deep learning project\model.pth")
torch.save(model.cpu().state_dict(), checkpoint_path)
print(f"Saved: {checkpoint_path}")
```

Run that cell in the same notebook session where `study.best_params` and the datasets exist. Once `model.pth` appears beside `app.py`, restart the server (`Ctrl+C`, then `uvicorn app:app --reload`). The app automatically loads the checkpoint at startup. If the file lives elsewhere, set `MODEL_PATH` to its full path **before** starting Uvicorn. The API expects the same architecture and a PyTorch state dict. Images are resized to 28 × 28 grayscale and normalized to the Fashion-MNIST pixel range before inference.

> The notebook's Optuna objective initializes an optimizer but fails to assign the optimizer selected by the trial. Its 89.08% result is the recorded notebook output, not a guaranteed benchmark for the corrected retraining recipe or new images.

> The displayed 89.08% is the notebook's recorded Optuna trial result, not a re-evaluation or a guarantee for new images. That trial's notebook code also initializes an optimizer without assigning the selected optimizer, so retraining should verify the optimizer wiring before treating the score as a reliable benchmark.
