from __future__ import annotations
from dataclasses import dataclass
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from .models import MLPSequenceClassifier, CausalTCNClassifier


@dataclass
class TrainResult:
    best_epoch: int
    train_loss: float
    valid_loss: float
    checkpoint: str | None
    device: str
    n_parameters: int
    validation_scores: np.ndarray


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_classifier(bundle, architecture="mlp", hidden_size=64, dropout=0.2,
                     learning_rate=1e-3, weight_decay=1e-4, batch_size=64,
                     max_epochs=40, patience=5, seed=42, checkpoint=None):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if architecture == "mlp":
        model = MLPSequenceClassifier(bundle.context_bars, len(bundle.columns), hidden_size, dropout)
    elif architecture in ("tcn", "cnn"):
        model = CausalTCNClassifier(len(bundle.columns), max(16, hidden_size // 2), dropout)
    else:
        raise ValueError("V10 smoke trainer supports mlp or tcn; GRU/LSTM/Transformer are future challengers.")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    train_ds = TensorDataset(torch.tensor(bundle.x_train), torch.tensor(bundle.y_train, dtype=torch.float32))
    loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    xv = torch.tensor(bundle.x_valid, device=device)
    yv = torch.tensor(bundle.y_valid, dtype=torch.float32, device=device)
    best, best_epoch, wait = float("inf"), -1, 0
    best_state = copy.deepcopy(model.state_dict())
    train_loss = float("nan")
    for epoch in range(max_epochs):
        model.train(); losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            valid_loss = float(criterion(model(xv), yv).cpu())
        if valid_loss < best - 1e-6:
            best, best_epoch, wait = valid_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(xv)).cpu().numpy()
    ckpt = None
    if checkpoint:
        path = Path(checkpoint); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "architecture": architecture, "columns": bundle.columns,
                    "context_bars": bundle.context_bars, "seed": seed}, path)
        ckpt = str(path)
    return TrainResult(best_epoch, train_loss, best, ckpt, str(device),
                       sum(p.numel() for p in model.parameters()), scores)


def smoke_train(bundle, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    result = train_classifier(bundle, max_epochs=3, patience=2, hidden_size=24,
                              checkpoint=out / "dl_smoke.pt")
    summary = {"status": "DEVELOPMENT ONLY", "best_epoch": result.best_epoch,
               "train_loss": result.train_loss, "valid_loss": result.valid_loss,
               "device": result.device, "n_parameters": result.n_parameters,
               "checkpoint": result.checkpoint,
               "note": "Smoke test only; not evidence of a tradable edge."}
    (out / "dl_smoke.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
