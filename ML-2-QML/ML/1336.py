import torch
from torch import nn
import pennylane as qml
import numpy as np

class ConvGen292(nn.Module):
    """Hybrid classical‑quantum convolution filter.

    The filter implements a 2‑D convolution followed by a sigmoid
    activation.  When *use_qc* is True an additional PennyLane
    variational circuit processes the flattened convolution output.
    The circuit is differentiable, allowing end‑to‑end training with
    back‑propagation.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 use_qc: bool = False,
                 qc_device: str = "default.qubit",
                 qc_layers: int = 2,
                 qc_params: list[float] | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.use_qc = use_qc
        if use_qc:
            self.qc_device = qml.device(qc_device, wires=kernel_size**2)
            init = np.random.uniform(0, np.pi, qc_layers * kernel_size**2)
            self.qc_params = nn.Parameter(torch.tensor(init, dtype=torch.float32))
            self.qc_layers = qc_layers
        else:
            self.qc_params = None

    def _quantum_circuit(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Variational ansatz used when *use_qc* is True."""
        wires = range(self.kernel_size**2)
        for layer in range(self.qc_layers):
            for w in wires:
                qml.RY(theta[layer * self.kernel_size**2 + w], wires=w)
            for w in wires:
                qml.CNOT(wires=(w, (w + 1) % len(wires)))
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the filter output."""
        conv_out = self.conv(x)
        activated = torch.sigmoid(conv_out - self.threshold)
        if self.use_qc:
            flat = activated.view(-1)
            qnode = qml.QNode(self._quantum_circuit,
                              self.qc_device,
                              interface="torch",
                              diff_method="backprop")
            return qnode(flat, self.qc_params)
        return activated

    def run(self, data) -> float:
        """Evaluate the filter on a single sample without gradients."""
        with torch.no_grad():
            inp = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            return float(self.forward(inp).item())

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} kernel={self.kernel_size} qc={self.use_qc}>"

def Conv() -> ConvGen292:
    """Drop‑in classical filter."""
    return ConvGen292(use_qc=False)

def ConvHybrid() -> ConvGen292:
    """Hybrid classical‑quantum filter."""
    return ConvGen292(use_qc=True)
