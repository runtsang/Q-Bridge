import torch
import pennylane as qml

class QuantumHybridClassifier:
    """Pure quantum implementation of a hybrid classifier using Pennylane.

    The circuit uses an entangled ansatz with RY rotations and CNOT entanglement,
    then measures PauliZ on the first qubit. The expectation value is mapped
    to a probability via a sigmoid function.
    """
    def __init__(self, n_qubits: int = 1, dev_name: str = "default.qubit", shots: int = 1024):
        self.dev = qml.device(dev_name, wires=n_qubits, shots=shots)
        self.n_qubits = n_qubits
        self._qnode = qml.qnode(self.dev, interface='torch', diff_method='backprop')(self._circuit)

    def _circuit(self, theta: torch.Tensor):
        for i in range(self.n_qubits):
            qml.RY(theta[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            features = features.unsqueeze(0)
        outputs = []
        for i in range(features.shape[0]):
            outputs.append(self._qnode(features[i]))
        probs = torch.sigmoid(torch.stack(outputs))
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumHybridClassifier"]
