"""MLP classifier heads on cached embeddings, many independent heads trained at once.

`HeadEnsemble` holds H independent copies of the original head (Linear->256->64->2, ReLU, dropout 0.3)
as batched weights. Each head sees only its own training rows, its own shuffles and minibatches
of `batch_size`, and the loss is the sum of the per-head mean losses, so with Adam (a per-parameter
optimiser) every head gets exactly the update it would get if trained alone.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = (256, 64)
DROPOUT = 0.3


class HeadEnsemble(nn.Module):
    def __init__(self, n_heads: int, in_dim: int, n_classes: int = 2):
        super().__init__()
        dims = (in_dim, *HIDDEN, n_classes)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            bound = 1 / math.sqrt(fan_in)  # same init as nn.Linear
            self.weights.append(nn.Parameter(torch.empty(n_heads, fan_in, fan_out).uniform_(-bound, bound)))
            self.biases.append(nn.Parameter(torch.empty(n_heads, 1, fan_out).uniform_(-bound, bound)))

    def forward(self, x):  # x: (H, batch, in_dim) -> (H, batch, n_classes)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.baddbmm(b, x, w)
            if i < len(self.weights) - 1:
                x = F.dropout(F.relu(x), DROPOUT, self.training)
        return x


def train_heads(X: torch.Tensor, rows: list[list[int]], labels: list[list[int]], *, n_classes: int = 2,
                batch_size: int = 32, n_epochs: int = 20, lr: float = 1e-3, seed: int = 0) -> HeadEnsemble:
    """Train one head per (rows, labels) training set; all sets must be the same length."""
    torch.manual_seed(seed)
    device = X.device
    rows_t = torch.tensor(rows, device=device)
    y_t = torch.tensor(labels, device=device)
    n_heads, n = rows_t.shape
    model = HeadEnsemble(n_heads, X.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_epochs):
        order = torch.argsort(torch.rand(n_heads, n, device=device), dim=1)
        for idx in order.split(batch_size, dim=1):
            xb = X[rows_t.gather(1, idx)]
            yb = y_t.gather(1, idx)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1), reduction="none")
            loss = loss.view(n_heads, -1).mean(dim=1).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.no_grad()
def predict_heads(model: HeadEnsemble, X: torch.Tensor, rows: list[list[int]]) -> list[np.ndarray]:
    """Class probabilities (n_rows, n_classes) for each head on its own evaluation rows."""
    probs = []
    for h, r in enumerate(rows):
        x = X[torch.tensor(r, device=X.device)].unsqueeze(0)
        for i, (w, b) in enumerate(zip(model.weights, model.biases)):
            x = torch.baddbmm(b[h:h + 1], x, w[h:h + 1])
            if i < len(model.weights) - 1:
                x = F.relu(x)
        probs.append(torch.softmax(x[0], dim=-1).cpu().numpy())
    return probs
