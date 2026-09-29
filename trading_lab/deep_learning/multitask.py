from __future__ import annotations
from dataclasses import dataclass
import copy
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from .models import MLPSequenceMultiTask, CausalTCNMultiTask
from .utility import CLASS_TARGETS, REG_TARGETS


@dataclass
class MultiTaskTrainResult:
    best_epoch: int
    train_loss: float
    valid_loss: float
    checkpoint: str | None
    device: str
    n_parameters: int
    validation_frame: object
    test_frame: object | None
    pos_weight: np.ndarray


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _decode(raw, bundle, index):
    import pandas as pd
    raw = np.asarray(raw)
    class_scores = 1.0 / (1.0 + np.exp(-np.clip(raw[:, :3], -30, 30)))
    reg = raw[:, 3:] * bundle.reg_std[None, :] + bundle.reg_mean[None, :]
    arr = np.concatenate([class_scores, reg], axis=1)
    return pd.DataFrame(arr, index=index, columns=list(CLASS_TARGETS) + list(REG_TARGETS))


def train_multitask(bundle, architecture="mlp", hidden_size=24, dropout=0.2,
                    learning_rate=1e-3, weight_decay=1e-4, batch_size=64,
                    max_epochs=30, patience=5, seed=42, checkpoint=None,
                    x_test=None, test_index=None):
    _seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if architecture == "mlp":
        model = MLPSequenceMultiTask(bundle.context_bars, len(bundle.columns), hidden_size, dropout)
    elif architecture in ("tcn", "cnn"):
        model = CausalTCNMultiTask(len(bundle.columns), max(16, hidden_size // 2), dropout)
    else:
        raise ValueError("Architecture utility supportée: mlp ou tcn.")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    positives = bundle.y_class_train.sum(axis=0)
    negatives = len(bundle.y_class_train) - positives
    # Train-only. Limite à 20 pour corriger la rareté sans laisser une poignée de positifs dominer le loss.
    pos_weight = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 20.0).astype(np.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="mean")
    huber = nn.SmoothL1Loss(reduction="mean")

    ds = TensorDataset(torch.tensor(bundle.x_train), torch.tensor(bundle.y_class_train), torch.tensor(bundle.y_reg_train))
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    xv = torch.tensor(bundle.x_valid, device=device)
    ycv = torch.tensor(bundle.y_class_valid, device=device)
    yrv = torch.tensor(bundle.y_reg_valid, device=device)

    best, best_epoch, wait = float("inf"), -1, 0
    best_state = copy.deepcopy(model.state_dict())
    last_train = float("nan")
    for epoch in range(int(max_epochs)):
        model.train(); losses = []
        for xb, yc, yr in loader:
            xb, yc, yr = xb.to(device), yc.to(device), yr.to(device)
            opt.zero_grad(); raw = model(xb)
            loss_cls = bce(raw[:, :3], yc)
            loss_reg = huber(raw[:, 3:], yr)
            loss = loss_cls + 0.35 * loss_reg
            loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        last_train = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            rawv = model(xv)
            vloss = float((bce(rawv[:, :3], ycv) + 0.35 * huber(rawv[:, 3:], yrv)).cpu())
        if vloss < best - 1e-6:
            best, best_epoch, wait = vloss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= int(patience): break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        rawv = model(xv).cpu().numpy()
        rawt = model(torch.tensor(x_test, device=device)).cpu().numpy() if x_test is not None and len(x_test) else None
    valid_frame = _decode(rawv, bundle, bundle.valid_index)
    test_frame = _decode(rawt, bundle, test_index) if rawt is not None else None
    ckpt = None
    if checkpoint:
        path = Path(checkpoint); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": best_state, "architecture": architecture, "columns": bundle.columns,
            "context_bars": bundle.context_bars, "seed": seed, "objective": "candidate_ranker_v11_3_2",
            "class_targets": list(CLASS_TARGETS), "reg_targets": list(REG_TARGETS),
            "reg_mean": bundle.reg_mean, "reg_std": bundle.reg_std,
            "scaler_mean": bundle.scaler.mean_, "scaler_scale": bundle.scaler.scale_,
            "fill_values": bundle.fill_values.to_numpy(dtype=float), "pos_weight": pos_weight,
        }, path)
        ckpt = str(path)
    return MultiTaskTrainResult(best_epoch, last_train, best, ckpt, str(device),
                                sum(p.numel() for p in model.parameters()), valid_frame, test_frame, pos_weight)
