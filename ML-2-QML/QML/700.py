import pennylane as qml
import numpy as np

class SamplerQNNExtended:
    """
    Variational sampler with parameter‑shift training.
    Parameters
    ----------
    qubits : int, default 2
        Number of qubits.
    layers : int, default 2
        Number of variational layers.
    dev : str or pennylane.Device, default 'default.qubit'
        Backend device.
    """
    def __init__(self, qubits: int = 2, layers: int = 2,
                 dev: str | qml.Device = "default.qubit") -> None:
        self.qubits = qubits
        self.layers = layers
        self.dev = qml.device(dev, wires=qubits)
        self.params = np.random.uniform(0, 2 * np.pi,
                                        size=(layers, qubits))
        self._build_qnode()

    def _variational_layer(self, params, wires):
        for i, w in enumerate(wires):
            qml.RZ(params[0, i], wires=w)
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, params):
            # Input encoding
            for i, w in enumerate(self.qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for layer in range(self.layers):
                self._variational_layer(params[layer], wires=range(self.qubits))
            # Measurement
            return qml.probs(wires=range(self.qubits))
        self.circuit = circuit

    def sample(self, inputs: np.ndarray, num_shots: int = 1024) -> dict:
        """Return a histogram of measurement outcomes."""
        probs = self.circuit(torch.tensor(inputs, dtype=torch.float32),
                             torch.tensor(self.params, dtype=torch.float32))
        counts = {format(i, f"0{self.qubits}b"): p.item()
                  for i, p in enumerate(probs)}
        # Convert probabilities to counts
        return {k: int(v * num_shots) for k, v in counts.items()}

    def train(self, data_loader, lr: float = 0.01, epochs: int = 10):
        """
        Very lightweight training loop using parameter‑shift gradients.
        data_loader yields (inputs, targets) tuples.
        """
        opt = qml.optimize.AdamOptimizer(lr)
        for _ in range(epochs):
            for inputs, targets in data_loader:
                def loss_fn(p):
                    probs = self.circuit(torch.tensor(inputs, dtype=torch.float32),
                                         torch.tensor(p, dtype=torch.float32))
                    # Cross‑entropy with one‑hot targets
                    target_probs = torch.tensor(targets, dtype=torch.float32)
                    return -torch.sum(target_probs * torch.log(probs + 1e-10)) / len(inputs)
                self.params = opt.step(loss_fn, self.params)

__all__ = ["SamplerQNNExtended"]
