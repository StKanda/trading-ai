from __future__ import annotations
from dataclasses import dataclass
import copy, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from .models import MLPSequenceMultiTask, CausalTCNMultiTask
from .economic import CLASS_TARGETS, REG_TARGETS

@dataclass
class EconomicTrainResult:
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
    ncls = len(CLASS_TARGETS)
    class_scores = 1.0 / (1.0 + np.exp(-np.clip(raw[:, :ncls], -30, 30)))
    reg = raw[:, ncls:] * bundle.reg_std[None, :] + bundle.reg_mean[None, :]
    return pd.DataFrame(np.concatenate([class_scores, reg], axis=1), index=index,
                        columns=list(CLASS_TARGETS) + list(REG_TARGETS))


def train_economic(bundle, architecture="mlp", hidden_size=24, dropout=0.2,
                   learning_rate=1e-3, weight_decay=1e-4, batch_size=64,
                   max_epochs=30, patience=5, seed=42, checkpoint=None,
                   x_test=None, test_index=None, progress_callback=None):
    _seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = len(CLASS_TARGETS) + len(REG_TARGETS)
    if architecture == "mlp":
        model = MLPSequenceMultiTask(bundle.context_bars, len(bundle.columns), hidden_size, dropout, outputs=outputs)
    elif architecture in ("tcn", "cnn"):
        model = CausalTCNMultiTask(len(bundle.columns), max(16, hidden_size // 2), dropout, outputs=outputs)
    else:
        raise ValueError("Architecture economic supportée: mlp ou tcn.")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positives = bundle.y_class_train.sum(axis=0)
    negatives = len(bundle.y_class_train) - positives
    pos_weight = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 20.0).astype(np.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="mean")
    huber = nn.SmoothL1Loss(reduction="mean")
    ds = TensorDataset(torch.tensor(bundle.x_train), torch.tensor(bundle.y_class_train), torch.tensor(bundle.y_reg_train))
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    xv = torch.tensor(bundle.x_valid, device=device)
    ycv = torch.tensor(bundle.y_class_valid, device=device)
    yrv = torch.tensor(bundle.y_reg_valid, device=device)
    ncls = len(CLASS_TARGETS)
    best, best_epoch, wait = float("inf"), -1, 0
    best_state = copy.deepcopy(model.state_dict()); last_train = float("nan")
    epochs_run = 0
    early_stopped = False
    for epoch in range(int(max_epochs)):
        model.train(); losses=[]
        for xb,yc,yr in loader:
            xb,yc,yr=xb.to(device),yc.to(device),yr.to(device)
            opt.zero_grad(); raw=model(xb)
            # Régression économique au centre du loss; classification = auxiliaire.
            loss_reg = huber(raw[:, ncls:], yr)
            loss_cls = bce(raw[:, :ncls], yc)
            loss = loss_reg + 0.35*loss_cls
            loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        last_train=float(np.mean(losses)); model.eval()
        with torch.no_grad():
            rawv=model(xv)
            vloss=float((huber(rawv[:, ncls:], yrv)+0.35*bce(rawv[:, :ncls], ycv)).cpu())
        if vloss < best - 1e-6:
            best,best_epoch,wait=vloss,epoch,0; best_state=copy.deepcopy(model.state_dict())
        else:
            wait += 1
        epochs_run = epoch + 1
        stopping = wait >= int(patience)
        if progress_callback is not None:
            progress_callback({"event": "epoch", "epoch": epochs_run, "max_epochs": int(max_epochs),
                               "best_epoch": best_epoch + 1, "early_stopping": stopping})
        if stopping:
            early_stopped = True
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        rawv=model(xv).cpu().numpy()
        rawt=model(torch.tensor(x_test, device=device)).cpu().numpy() if x_test is not None and len(x_test) else None
    valid_frame=_decode(rawv,bundle,bundle.valid_index)
    test_frame=_decode(rawt,bundle,test_index) if rawt is not None else None
    ckpt=None
    if checkpoint:
        path=Path(checkpoint); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":best_state,"architecture":architecture,"columns":bundle.columns,
            "context_bars":bundle.context_bars,"seed":seed,"objective":"direct_economic_v11_4",
            "class_targets":list(CLASS_TARGETS),"reg_targets":list(REG_TARGETS),
            "reg_mean":bundle.reg_mean,"reg_std":bundle.reg_std,
            "scaler_mean":bundle.scaler.mean_,"scaler_scale":bundle.scaler.scale_,
            "fill_values":bundle.fill_values.to_numpy(dtype=float),"pos_weight":pos_weight,
        },path); ckpt=str(path)
    if progress_callback is not None:
        progress_callback({"event": "finished", "epochs_run": epochs_run,
                           "best_epoch": best_epoch + 1, "early_stopped": early_stopped})
    return EconomicTrainResult(best_epoch,last_train,best,ckpt,str(device),
                               sum(p.numel() for p in model.parameters()),valid_frame,test_frame,pos_weight)
