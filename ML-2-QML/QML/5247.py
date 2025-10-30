import torch
import torch.nn as nn
import torch.quantum as tq
from torchquantum import QuantumDevice
from torchquantum.functional import func_name_dict


class QuantumFilter(tq.QuantumModule):
    """Variational filter that mirrors the classical 2×2 convolution but uses a quantum circuit."""
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
        )
        self.var_layer = tq.RandomLayer(n_ops=4 * n_layers, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: QuantumDevice, data: torch.Tensor) -> torch.Tensor:
        qdev.reset_states(data.shape[0])
        for _ in range(self.n_layers):
            self.encoder(qdev, data)
            self.var_layer(qdev)
        return self.measure(qdev).view(-1, 4)


class QuantumQuanvolutionHybrid(nn.Module):
    """Quantum‑only version of the hybrid architecture that replaces the classical conv with quantum patches."""
    def __init__(self, kernel_size: int = 2, n_qubits: int = 4, n_layers: int = 2, num_classes: int = 10):
        super().__init__()
        self.filter = QuantumFilter(n_qubits=n_qubits, n_layers=n_layers)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # slice image into 2×2 patches and run the quantum filter on each
        patches = x.view(x.size(0), 28, 28)
        batch_size = x.size(0)
        qdev = QuantumDevice(self.filter.n_qubits, bsz=batch_size)
        outputs = []
        for i in range(28 * 28):
            patch = patches[:, i // 28, i % 28]
            outputs.append(self.filter(qdev, patch))
        out = torch.cat(outputs, dim=1)
        logits = self.linear(out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuantumFilter", "QuantumQuanvolutionHybrid"]
