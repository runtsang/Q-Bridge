import torch
import torchquantum as tq
import numpy as np
from torchquantum.functional import func_name_dict
from typing import Sequence

class HybridKernelQCNN(tq.QuantumModule):
    """Quantum QCNN-inspired variational circuit."""
    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)
        self.ansatz_desc = self._build_ansatz_desc()

    def _build_ansatz_desc(self):
        desc = []
        # Feature map: ry on each qubit
        for i in range(self.n_qubits):
            desc.append({"func": "ry", "wires": [i], "input_idx": [i]})
        # Convolution and pooling layers
        for layer in range(self.n_layers):
            # Convolution: entangle pairs
            for i in range(0, self.n_qubits, 2):
                desc.append({"func": "cx", "wires": [i, i+1], "input_idx": []})
                desc.append({"func": "ry", "wires": [i], "input_idx": [i]})
                desc.append({"func": "ry", "wires": [i+1], "input_idx": [i+1]})
            # Pooling: entangle and discard
            for i in range(0, self.n_qubits, 4):
                desc.append({"func": "cx", "wires": [i, i+2], "input_idx": []})
        return desc

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.ansatz_desc:
            params = x[:, info["input_idx"]] if info["input_idx"] else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Run the circuit on input x and return |<0|ψ_x>|."""
        self.forward(self.q_device, x)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute quantum kernel matrix via overlap."""
        kernel = np.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                # Encode x
                self.forward(self.q_device, x)
                psi_x = self.q_device.states.clone()
                # Encode y
                self.forward(self.q_device, y)
                psi_y = self.q_device.states.clone()
                overlap = torch.abs(torch.dot(psi_x.conj(), psi_y))**2
                kernel[i, j] = overlap.item()
        return kernel

__all__ = ["HybridKernelQCNN"]
