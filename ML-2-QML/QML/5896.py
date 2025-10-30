"""Quantum kernel using Pennylane with a tunable parameterised circuit."""

import pennylane as qml
import torch
from torch import nn
from typing import Sequence, Optional

class QuantumKernelMethod(nn.Module):
    """
    Quantum kernel implemented with Pennylane.
    The circuit encodes classical data into a quantum state.
    The kernel is the absolute overlap between two states prepared
    from inputs x and y.
    """

    def __init__(self,
                 n_qubits: int = 4,
                 layers: int = 2,
                 device_name: str = 'default.qubit',
                 wires: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.device = qml.device(device_name, wires=n_qubits)
        self.wires = wires or list(range(n_qubits))
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device, interface='torch')
        def circuit(params, x, y=None):
            # encode x
            for i, w in enumerate(self.wires):
                qml.RY(x[i], wires=w)
            # apply parameterised layers
            for _ in range(self.layers):
                for i in self.wires:
                    qml.RX(params[i], wires=i)
                for i in range(len(self.wires) - 1):
                    qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
            # encode y with negative sign for inner product
            if y is not None:
                for i, w in enumerate(self.wires):
                    qml.RY(-y[i], wires=w)
                # reapply layers to make a symmetric kernel
                for _ in range(self.layers):
                    for i in self.wires:
                        qml.RX(-params[i], wires=i)
                    for i in range(len(self.wires) - 1):
                        qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
            return qml.expval(qml.PauliZ(self.wires[0]))

        self.circuit = circuit
        # Initialize parameters
        self.params = nn.Parameter(torch.randn(self.n_qubits, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel between two input vectors.
        """
        x = x.to(self.params.device)
        y = y.to(self.params.device)
        return torch.abs(self.circuit(self.params, x, y))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Compute the Gram matrix between two collections of vectors.
        """
        a_stack = torch.stack([torch.tensor(v, dtype=torch.float32) for v in a])
        b_stack = torch.stack([torch.tensor(v, dtype=torch.float32) for v in b])
        k_mat = torch.zeros((len(a), len(b)), dtype=torch.float32)
        for i, x in enumerate(a_stack):
            for j, y in enumerate(b_stack):
                k_mat[i, j] = self.forward(x, y)
        return k_mat

    def __repr__(self) -> str:
        return f"QuantumKernelMethod(n_qubits={self.n_qubits}, layers={self.layers})"

__all__ = ["QuantumKernelMethod"]
