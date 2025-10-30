import pennylane as qml
import torch
import numpy as np

class EstimatorQNN(torch.nn.Module):
    """
    Variational quantum neural network implemented with Pennylane.
    Supports 2‑qubit circuits with multiple parameterised layers and
    automatic differentiation via the torch interface.
    """
    def __init__(self,
                 n_qubits: int = 2,
                 n_layers: int = 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Parameters: [layer, qubit, rotation angles]
        self.weights = torch.nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.qnode = qml.qnode(self._circuit,
                               interface="torch",
                               diff_method="backprop")

    def _circuit(self, inputs: torch.Tensor, weights: torch.Tensor):
        # Feature encoding via RX
        for i in range(self.n_qubits):
            qml.RX(inputs[i], wires=i)
        # Variational layers
        for l in range(self.n_layers):
            for w, i in zip(weights[l], range(self.n_qubits)):
                qml.Rot(*w, wires=i)
            # Small entangling block
            qml.CNOT(wires=[0, 1])
        # Single‑qubit expectation as output
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expectation value is a scalar in [−1, 1]
        return self.qnode(x, self.weights)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)

__all__ = ["EstimatorQNN"]
