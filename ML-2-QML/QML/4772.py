import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import cnot
from typing import Sequence

class ConvLayer(tq.QuantumModule):
    """Convolutional layer inspired by QCNN."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, inverse: bool = False) -> None:
        params = -x if inverse else x
        for i in range(self.n_wires // 2):
            w1, w2 = 2 * i, 2 * i + 1
            tq.rz(q_device, wires=[w2], params=-np.pi / 2)
            tq.cx(q_device, wires=[w2, w1])
            tq.rz(q_device, wires=[w1], params=params[:, w1])
            tq.ry(q_device, wires=[w2], params=params[:, w2])
            tq.cx(q_device, wires=[w1, w2])
            tq.ry(q_device, wires=[w2], params=params[:, w2])
            tq.cx(q_device, wires=[w2, w1])
            tq.rz(q_device, wires=[w1], params=np.pi / 2)

class PoolLayer(tq.QuantumModule):
    """Pooling layer inspired by QCNN."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, inverse: bool = False) -> None:
        params = -x if inverse else x
        for i in range(self.n_wires // 2):
            w1, w2 = 2 * i, 2 * i + 1
            tq.rz(q_device, wires=[w2], params=-np.pi / 2)
            tq.cx(q_device, wires=[w2, w1])
            tq.rz(q_device, wires=[w1], params=params[:, w1])
            tq.ry(q_device, wires=[w2], params=params[:, w2])
            tq.cx(q_device, wires=[w1, w2])
            tq.ry(q_device, wires=[w2], params=params[:, w2])

class QLSTMLayer(tq.QuantumModule):
    """Quantum LSTM cell for feature enhancement."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, q_device: tq.QuantumDevice) -> None:
        self.encoder(q_device)
        for wire, gate in enumerate(self.params):
            gate(q_device, wires=[wire])
        for wire in range(self.n_wires):
            if wire == self.n_wires - 1:
                cnot(q_device, wires=[wire, 0])
            else:
                cnot(q_device, wires=[wire, wire + 1])

class QuantumKernelMethod(tq.QuantumModule):
    """
    Quantum kernel that uses a quantum convolutional‑pooling ansatz
    derived from QCNN and optionally a quantum LSTM for feature
    transformation. Mirrors the classical implementation but leverages
    quantum feature maps.
    """
    def __init__(self,
                 n_qubits: int = 4,
                 use_qlstm: bool = False) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.feature_map = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
        )
        self.conv = ConvLayer(n_qubits)
        self.pool = PoolLayer(n_qubits)
        self.qlstm = QLSTMLayer(n_qubits) if use_qlstm else None
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the overlap kernel between two batches of vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            Shape (N, n_qubits).  They are encoded into the quantum state
            and the overlap of the resulting states is returned.
        """
        x = x.view(-1, self.n_qubits)
        y = y.view(-1, self.n_qubits)

        # Encode x
        self.q_device.reset_states(x.shape[0])
        self.feature_map(self.q_device, x)
        self.conv(self.q_device, x)
        self.pool(self.q_device, x)
        if self.qlstm:
            self.qlstm(self.q_device)

        # Encode y with inverse operations to obtain overlap
        self.feature_map(self.q_device, y, inverse=True)
        self.pool(self.q_device, y, inverse=True)
        self.conv(self.q_device, y, inverse=True)

        # Overlap measurement
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  n_qubits: int = 4,
                  use_qlstm: bool = False) -> np.ndarray:
    """Utility to compute Gram matrix from a list of tensors."""
    kernel = QuantumKernelMethod(n_qubits=n_qubits, use_qlstm=use_qlstm)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QuantumKernelMethod", "kernel_matrix"]
