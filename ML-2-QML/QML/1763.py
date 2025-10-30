import pennylane as qml
import torch
import numpy as np
from typing import Sequence


class QuantumKernel:
    """
    Variational quantum kernel implemented with PennyLane.

    Parameters
    ----------
    n_qubits : int, default=4
        Number of qubits in the circuit.
    n_layers : int, default=2
        Depth of the variational ansatz.
    dev_name : str, default="default.qubit"
        PennyLane backend device name.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2,
                 dev_name: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(dev_name, wires=n_qubits)

        # Trainable parameters for the variational part
        self.params = torch.randn(n_layers * n_qubits, requires_grad=True)

        # Define the QNode once; it will use the current params
        def _circuit(x, params):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(x[i], wires=i)
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RZ(params[l * self.n_qubits + i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.state()

        self.circuit = qml.QNode(_circuit, self.dev, interface="torch")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value between two feature vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of shape (n_qubits,).  The method also accepts
            batched inputs of shape (batch, n_qubits).

        Returns
        -------
        torch.Tensor
            Kernel value(s) of shape (batch_x, batch_y).
        """
        # Ensure inputs are 2‑D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        psi_x = self.circuit(x, self.params)  # (batch_x, 2**n_qubits)
        psi_y = self.circuit(y, self.params)  # (batch_y, 2**n_qubits)

        # Inner product (fidelity) between state vectors
        prod = torch.sum(torch.conj(psi_x).unsqueeze(1) * psi_y.unsqueeze(0), dim=-1)
        return torch.abs(prod)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Convenience wrapper to compute the Gram matrix between two lists.

        Parameters
        ----------
        a : Sequence[torch.Tensor]
            First list of feature vectors.
        b : Sequence[torch.Tensor]
            Second list of feature vectors.

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        mat = torch.stack([self.forward(x, y) for x in a for y in b])
        mat = mat.view(len(a), len(b))
        return mat.detach().cpu().numpy()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  kernel: QuantumKernel) -> np.ndarray:
    """
    Compute the Gram matrix between two lists of tensors using the given quantum kernel.

    Parameters
    ----------
    a : Sequence[torch.Tensor]
        First list of input vectors.
    b : Sequence[torch.Tensor]
        Second list of input vectors.
    kernel : QuantumKernel
        Instance of the variational quantum kernel.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    mat = torch.stack([kernel.forward(x, y) for x in a for y in b])
    mat = mat.view(len(a), len(b))
    return mat.detach().cpu().numpy()


__all__ = ["QuantumKernel", "kernel_matrix"]
