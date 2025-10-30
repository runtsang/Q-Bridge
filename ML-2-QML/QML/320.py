import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionFilter(tq.QuantumModule):
    """Parameterised quanvolution filter with a variational layer."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.var_layer = tq.ParameterizedLayer(
            n_ops=4,
            n_wires=self.n_wires,
            ops=[tq.RZ] * 4,
            wires=list(range(self.n_wires))
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.var_layer(qdev)
                qdev.cnot(0, 1)
                qdev.cnot(2, 3)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the quantum quanvolutional filter followed by a linear head."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

    def fit(self, dataloader, epochs=5, lr=1e-3, device="cpu"):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logp = self.forward(xb)
                loss = F.nll_loss(logp, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(dataloader.dataset)
            print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    def evaluate(self, dataloader, device="cpu"):
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                logp = self.forward(xb)
                preds = logp.argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
