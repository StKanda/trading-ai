from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class MLPSequenceClassifier(nn.Module):
    def __init__(self, context_bars, n_features, hidden_size=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(context_bars * n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_size // 2), 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_pad, 0)))


class CausalTCNClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=32, dropout=0.2):
        super().__init__()
        self.tcn = nn.Sequential(
            CausalConv1d(n_features, hidden_size, 3, 1), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(hidden_size, hidden_size, 3, 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        z = self.tcn(x.transpose(1, 2))
        return self.head(z[:, :, -1]).squeeze(-1)


class MLPSequenceMultiTask(nn.Module):
    """Petit réseau multi-tâches : 3 logits target-before-stop + 5 régressions."""
    def __init__(self, context_bars, n_features, hidden_size=64, dropout=0.2, outputs=8):
        super().__init__()
        mid = max(16, hidden_size // 2)
        self.body = nn.Sequential(
            nn.Flatten(), nn.Linear(context_bars * n_features, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, mid), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(mid, outputs)

    def forward(self, x):
        return self.head(self.body(x))


class CausalTCNMultiTask(nn.Module):
    def __init__(self, n_features, hidden_size=32, dropout=0.2, outputs=8):
        super().__init__()
        self.tcn = nn.Sequential(
            CausalConv1d(n_features, hidden_size, 3, 1), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(hidden_size, hidden_size, 3, 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_size, outputs)

    def forward(self, x):
        z = self.tcn(x.transpose(1, 2))
        return self.head(z[:, :, -1])
