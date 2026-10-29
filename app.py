"""Fashion MNIST inference API for the project dashboard."""

from __future__ import annotations

import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pydantic import BaseModel


ROOT = Path(__file__).parent
WEIGHTS = Path(os.environ.get("MODEL_PATH", ROOT / "model.pth"))
LABELS = [
    "T-shirt / top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class FashionANN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        input_size = 784
        for _ in range(4):
            layers.extend([
                torch.nn.Linear(input_size, 88),
                torch.nn.BatchNorm1d(88),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
            ])
            input_size = 88
        layers.append(torch.nn.Linear(88, 10))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


app = FastAPI(title="Threadline · Fashion Classifier", version="1.0.0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: FashionANN | None = None
model_error: str | None = None
if WEIGHTS.exists():
    try:
        model = FashionANN().to(device)
        state = torch.load(WEIGHTS, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
    except Exception as exc:  # expose a helpful setup message in the UI
        model_error = f"Could not load {WEIGHTS.name}: {exc}"
        model = None


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str
    accuracy: float
    checkpoint: str
    message: str | None = None


@app.get("/api/health", response_model=Health)
def health() -> Health:
    return Health(
        status="ready" if model else "setup_required",
        model_loaded=model is not None,
        model_name="Fashion ANN · Optuna best trial",
        device=str(device).upper(),
        accuracy=88.77,
        checkpoint=WEIGHTS.name,
        message=model_error or (None if model else "Add the trained model.pth checkpoint to the project folder."),
    )


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model checkpoint is not loaded. Add model.pth and restart the server.")
    if file.content_type not in {"image/png", "image/jpeg", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPG, WEBP, or BMP image.")
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be smaller than 8 MB.")
    try:
        image = Image.open(io.BytesIO(data)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="This image could not be opened.") from exc
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
    pixels = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(1, 784) / 255.0
    with torch.inference_mode():
        probabilities = torch.softmax(model(pixels.to(device)), dim=1)[0].cpu()
    top = torch.topk(probabilities, 3)
    return {
        "prediction": LABELS[int(top.indices[0])],
        "class_id": int(top.indices[0]),
        "confidence": round(float(top.values[0]) * 100, 2),
        "top_predictions": [
            {"label": LABELS[int(idx)], "class_id": int(idx), "confidence": round(float(score) * 100, 2)}
            for idx, score in zip(top.indices, top.values)
        ],
        "device": str(device).upper(),
    }


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/{asset:path}", include_in_schema=False)
def assets(asset: str) -> FileResponse:
    path = (ROOT / asset).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path)
