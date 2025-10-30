import pennylane as qml
import pennylane.numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

def feature_map(x: np.ndarray) -> Callable[[int], None]:
    def circuit(q: int):
        qml.AngleEmbedding(x, wires=list(range(q)), rotation="Y")
    return circuit

def ansatz(params: np.ndarray, num_qubits: int) -> Callable[[int], None]:
    def circuit(q: int):
        for i in range(num_qubits):
            qml.RY(params[i], wires=i)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        for i in range(num_qubits):
            qml.RY(params[i + num_qubits], wires=i)
    return circuit

@dataclass
class QCNNModel:
    """Hybrid QCNN implemented as a Pennylane QNode with a feature map, ansatz, and expectation‑value measurement."""
    num_qubits: int = 8
    param_shape: Tuple[int,...] = (16,)

    def __post_init__(self) -> None:
        self.device = qml.device("default.qubit", wires=self.num_qubits)
        self.params = np.random.randn(*self.param_shape)

    def qnode(self, x: np.ndarray) -> np.ndarray:
        @qml.qnode(self.device, interface="autograd")
        def circuit():
            feature_map(x)(self.num_qubits)
            ansatz(self.params, self.num_qubits)(self.num_qubits)
            return qml.expval(qml.PauliZ(wires=range(self.num_qubits)))
        return circuit()

    def forward(self, x: np.ndarray) -> np.ndarray:
        val = self.qnode(x)
        return 1 / (1 + np.exp(-val))

def QCNN() -> QCNNModel:
    """Factory creating a fully‑initialized QCNNModel."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
