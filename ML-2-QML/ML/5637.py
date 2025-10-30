import torch
from torch import nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    """A lightweight 1‑D convolution block with optional residual."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, residual=False):
        super().__init__()
        self.residual = residual
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride, padding=kernel//2)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        if residual and in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        return y + self.skip(x)

class PoolingLayer(nn.Module):
    """Multi‑scale pooling followed by a flatten‑and‑dense step."""
    def __init__(self, in_dim, pool_sizes=[2,4,5]):
        super().__init__()
        self.pool = nn.ModuleList(
            [nn.MaxPool1d(p, stride=p) for p in pool_sizes]
        )
        self.fc   = nn.Linear(in_dim * len(pool_sizes), 16)

    def forward(self, x):
        pools = [p(x) for p in self.pool]
        out   = torch.cat([p_.view(-1, p_.size(1)) for p_ in pools], dim=1)
        return self.fc(out)

class QCNNModel(nn.Module):
    """Hybrid classical‑quantum CNN for 1‑D signal classification."""
    def __init__(self, input_dim=8, num_classes=1):
        super().__init__()
        # Classical backbone
        self.feature_map = nn.Sequential(
            ConvBlock(1, 4, residual=True),
            ConvBlock(4, 8, residual=True),
            ConvBlock(8, 8, kernel=1)
        )
        self.pool = PoolingLayer(8)
        self.classifier = nn.Sequential(
            nn.Linear(16, 32),
            nn.Linear(32, num_classes)
        )

        # Quantum component
        self.qc = None  # placeholder for a Pennylane QNode; set via set_qnode

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = x.unsqueeze(1)   # add channel dimension
        x = self.feature_map(x)
        x = self.pool(x)
        if self.qc is not None:
            # detach for quantum circuit, convert to numpy
            q_in = x.detach().cpu().numpy()
            q_out = self.qc(q_in)
            # embed quantum output as feature vector
            q_feat = torch.tensor(q_out, device=x.device, dtype=torch.float32)
            x = torch.cat([x, q_feat], dim=1)
        return self.classifier(x)

    def set_qnode(self, qnode):
        """Attach a quantum node to the model."""
        self.qc = qnode

__all__ = ["QCNNModel"]
