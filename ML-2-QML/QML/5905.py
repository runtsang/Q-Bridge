import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionFilter(nn.Module):
    """
    Quantum filter applying a variational circuit to each 2x2 image patch.
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 2,
                 device: qml.Device | None = None, seed: int = 42,
                 freeze: bool = False):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.freeze = freeze
        self.device = device or qml.device("default.qubit", wires=n_wires)
        self.params = nn.Parameter(torch.randn(n_layers, n_wires, 3))
        if self.freeze:
            self.params.requires_grad = False
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, x: torch.Tensor) -> list[torch.Tensor]:
        for wire in range(self.n_wires):
            qml.RY(x[wire], wires=wire)
        for layer in range(self.n_layers):
            for wire in range(self.n_wires):
                qml.RX(self.params[layer, wire, 0], wires=wire)
                qml.RY(self.params[layer, wire, 1], wires=wire)
                qml.RZ(self.params[layer, wire, 2], wires=wire)
            for wire in range(self.n_wires - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CNOT(wires=[self.n_wires - 1, 0])
        return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2]
                patch_flat = patch.view(B, -1)
                expvals = torch.stack([self.qnode(patch_flat[i]) for i in range(B)], dim=0)
                patches.append(expvals)
        features = torch.cat(patches, dim=1)
        return features

    def set_pretrained(self, state_dict: dict) -> None:
        self.load_state_dict(state_dict, strict=False)

    def freeze_filter(self) -> None:
        self.freeze = True
        self.params.requires_grad = False

    def unfreeze_filter(self) -> None:
        self.freeze = False
        self.params.requires_grad = True

class QuanvolutionClassifier(nn.Module):
    """
    Hybrid network using the quantum quanvolution filter followed by a linear head.
    """
    def __init__(self, num_classes: int = 10, filter_kwargs: dict | None = None,
                 head_kwargs: dict | None = None):
        super().__init__()
        if filter_kwargs is None:
            filter_kwargs = {}
        self.qfilter = QuanvolutionFilter(**filter_kwargs)
        dummy = torch.zeros(1, 1, 28, 28)
        n_features = self.qfilter(dummy).shape[1]
        if head_kwargs is None:
            head_kwargs = {}
        self.linear = nn.Linear(n_features, num_classes, **head_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)
