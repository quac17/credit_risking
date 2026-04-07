from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TabularDeepCreditNet(nn.Module):
    """
    Embedding (từng cột phân loại) + vector số → Residual MLP.
    - head='softmax': logits (B, K)
    - head='coral': logits (B, K-1) cho CORAL (BCE theo từng ngưỡng tích lũy)
    """

    def __init__(
        self,
        cat_cardinalities: list[int],
        num_numeric: int,
        emb_dim: int,
        hidden_dims: tuple[int, ...],
        num_classes: int,
        dropout: float,
        head: str = "softmax",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.head_type = head
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cat_cardinalities])
        in_dim = num_numeric + len(cat_cardinalities) * emb_dim
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dims[-1], dropout) for _ in range(2)])
        out_dim = num_classes if head == "softmax" else num_classes - 1
        self.head = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_cat: (B, n_cat) int64 indices
        """
        embs = []
        for i, emb in enumerate(self.embs):
            embs.append(emb(x_cat[:, i]))
        h = torch.cat([x_num] + embs, dim=1)
        h = self.mlp(h)
        for blk in self.res_blocks:
            h = blk(h)
        return self.head(h)


def coral_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, K-1)
    targets: (B,) trong {0,...,K-1}
    Mục tiêu j: P(y > j) — nhãn nhị phân cho từng j.
    """
    k = logits.size(1)
    device = logits.device
    batch = logits.size(0)
    loss = 0.0
    for j in range(k):
        y_j = (targets > j).float()
        loss = loss + nn.functional.binary_cross_entropy_with_logits(logits[:, j], y_j)
    return loss / k


@torch.no_grad()
def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """
    logits (B, K-1): P(y > j) = sigmoid(z_j), j = 0..K-2.
    P(y=0)=1-P(y>0); P(y=c)=P(y>c-1)-P(y>c); P(y=K-1)=P(y>K-2).
    """
    probs = torch.sigmoid(logits)
    k = probs.size(1) + 1
    batch = logits.size(0)
    device = logits.device
    p_class = torch.zeros(batch, k, device=device)
    p_class[:, 0] = 1.0 - probs[:, 0]
    for c in range(1, k - 1):
        p_class[:, c] = probs[:, c - 1] - probs[:, c]
    p_class[:, k - 1] = probs[:, k - 2]
    return p_class.argmax(dim=1)
