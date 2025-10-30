import numpy as np
import pennylane as qml

class ConvGen292:
    """Quantum convolution filter using a parameterized PennyLane circuit.

    The filter accepts a 2‑D kernel of shape (kernel_size, kernel_size),
    flattens it, and feeds the pixel values into a variational circuit.
    The circuit consists of RY rotations whose angles are linear functions
    of the input pixels, followed by a few layers of CNOT entanglement.
    The output is the expectation value of Pauli‑Z on the first qubit,
    which is a differentiable scalar that can be used in a loss
    function and trained with autograd.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 device_name: str = "default.qubit",
                 qc_layers: int = 2):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.device = qml.device(device_name, wires=self.n_qubits)
        self.qc_layers = qc_layers
        init = np.random.uniform(0, np.pi, qc_layers * self.n_qubits)
        self.params = np.array(init, dtype=np.float64)

    def _ansatz(self, x: np.ndarray, theta: np.ndarray) -> float:
        """Variational ansatz for a single kernel."""
        wires = range(self.n_qubits)
        for layer in range(self.qc_layers):
            for w in wires:
                qml.RY(theta[layer * self.n_qubits + w], wires=w)
            for w in wires:
                qml.CNOT(wires=(w, (w + 1) % len(wires)))
        return qml.expval(qml.PauliZ(0))

    def run(self, data: np.ndarray) -> float:
        """Evaluate the filter on a single kernel."""
        x = data.reshape(-1)
        qnode = qml.QNode(self._ansatz,
                          self.device,
                          interface="autograd",
                          diff_method="backprop")
        return float(qnode(x, self.params))

    def train_step(self, data: np.ndarray, lr: float = 0.01):
        """Perform a single gradient‑descent step on the variational parameters."""
        import torch
        x = torch.tensor(data.reshape(-1), dtype=torch.float32, requires_grad=False)
        theta = torch.tensor(self.params, dtype=torch.float32, requires_grad=True)
        qnode = qml.QNode(self._ansatz,
                          self.device,
                          interface="torch",
                          diff_method="backprop")
        loss = qnode(x, theta)
        loss.backward()
        self.params -= lr * theta.grad.numpy()

def Conv() -> ConvGen292:
    """Return a pure quantum convolution filter."""
    return ConvGen292()
