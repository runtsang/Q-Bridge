import pennylane as qml
import torch
from torch import nn
import numpy as np

class QCNNModel(nn.Module):
    """
    Hybrid quantum‑classical QCNN using PennyLane’s TorchLayer.
    Feature map: Hadamard + RZ per qubit.
    Ansatz: 3‑parameter RX,RZ,RZ rotations per qubit, followed by cyclic CNOT entanglement.
    Observable: Pauli‑Z on qubit 0.
    """
    def __init__(self,
                 n_qubits: int = 8,
                 n_layers: int = 3,
                 dropout: float | None = None,
                 seed: int = 1234) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed
        np.random.seed(seed)

        # Device with shot‑based estimation for gradient estimation
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=1024)

        self._circuit = self._build_circuit()
        self.qnn = qml.qnn.TorchLayer(self._circuit,
                                      weight_shapes={"weights": (n_layers, n_qubits, 3)},
                                      interface="torch")

        self.head = nn.Linear(1, 1)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Feature map
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(inputs[i], wires=i)

            # Ansatz layers
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                    qml.RZ(weights[l, i, 2], wires=i)
                # Cyclic CNOT entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])

            return qml.expval(qml.PauliZ(0))
        return circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, n_qubits).
        Returns
        -------
        torch.Tensor
            Probabilities in [0, 1], shape (batch, 1).
        """
        logits = self.qnn(inputs)
        if self.dropout:
            logits = self.dropout(logits)
        out = self.head(logits)
        return torch.sigmoid(out)

def QCNN() -> QCNNModel:
    """Factory returning a fully‑initialized hybrid QCNN."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
