import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridClassifier(nn.Module):
    '''Variational quantum circuit head for binary classification.

    The network attaches a Pennylane variational ansatz to the
    classical backbone.  The parameters are trained jointly
    with the rest of the network using Pennylane's autograd
    interface.
    '''
    def __init__(self, n_qubits: int = 1, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Parameters: (n_layers, 3) for RX, RY, RZ angles
        self.params = nn.Parameter(torch.randn(n_layers, 3))

        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, params: torch.Tensor):
            qml.RX(x, wires=0)
            for i in range(self.n_layers):
                qml.Rot(*params[i], wires=0)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x shape (batch, 1) or (batch,)
        if x.dim() == 2 and x.size(1) == 1:
            x = x.squeeze(-1)
        out = self.circuit(x, self.params)
        probs = torch.sigmoid(out)
        return torch.cat([probs, 1 - probs], dim=-1)

    def evaluate(self, dataloader, device='cpu'):
        '''Compute ROC‑AUC and calibration curve on a dataset.'''
        from sklearn.metrics import roc_auc_score, brier_score_loss
        self.eval()
        probs, labels = [], []
        with torch.no_grad():
            for batch, target in dataloader:
                batch = batch.to(device)
                target = target.to(device)
                out = self(batch)
                probs.append(out[:, 0].cpu())
                labels.append(target.cpu())
        probs = torch.cat(probs).numpy()
        labels = torch.cat(labels).numpy()
        auc = roc_auc_score(labels, probs)
        calib = brier_score_loss(labels, probs)
        return {'auc': auc, 'brier': calib}

__all__ = ['HybridClassifier']
