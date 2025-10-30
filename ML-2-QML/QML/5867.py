import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np
from typing import Sequence

class QCNNAnsatzQuantum(tq.QuantumModule):
    """
    Variational ansatz that mimics a QCNN: a sequence of two‑qubit
    convolution and pooling blocks encoded as a flat list of gate specs.
    """
    def __init__(self, param_list: Sequence[dict]) -> None:
        super().__init__()
        self.param_list = param_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode input data
        for info in self.param_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Encode negative of the second sample (to compute overlap)
        for info in reversed(self.param_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QCNNQuantumKernel(tq.QuantumModule):
    """
    Quantum kernel that evaluates the overlap between two data encodings
    using a QCNN‑style ansatz.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QCNNAnsatzQuantum(self._build_ansatz())

    def _build_ansatz(self) -> list[dict]:
        """
        Construct a minimal QCNN‑inspired ansatz.
        Each pair of qubits receives an Ry rotation followed by a CX,
        repeated for each convolutional layer.
        """
        param_list: list[dict] = []
        p = 0
        for i in range(0, self.n_wires, 2):
            # Convolution block (simple 2‑qubit rotation pattern)
            param_list.append({"input_idx": [p], "func": "ry", "wires": [i]})
            param_list.append({"input_idx": [p + 1], "func": "ry", "wires": [i + 1]})
            param_list.append({"input_idx": [p + 2], "func": "cx", "wires": [i, i + 1]})
            p += 3
        # Pooling block: discard one qubit via a CX and a rotation
        param_list.append({"input_idx": [p], "func": "ry", "wires": [0]})
        param_list.append({"input_idx": [p + 1], "func": "cx", "wires": [0, 1]})
        return param_list

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return the absolute value of the amplitude of the |0...0> state
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Compute the Gram matrix using the QCNN‑inspired quantum kernel.
    """
    kernel = QCNNQuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QCNNAnsatzQuantum", "QCNNQuantumKernel", "kernel_matrix"]
