"""Quantum sampler network using Pennylane with a variational circuit."""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

class SamplerQNN:
    """Variational quantum sampler with two qubits and parameterized rotation gates."""
    def __init__(self, dev: qml.Device | None = None, num_qubits: int = 2, hidden_layers: int = 2):
        self.num_qubits = num_qubits
        self.hidden_layers = hidden_layers
        self.dev = dev or qml.device("default.qubit", wires=num_qubits, shots=8192)
        # Parameter shapes: inputs (2) + weights (hidden_layers * 4)
        self.input_params = np.random.uniform(0, 2*np.pi, (2,))
        self.weight_params = np.random.uniform(0, 2*np.pi, (hidden_layers, 4))
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray):
            # Encode inputs
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            # Variational layers
            for layer in range(self.hidden_layers):
                qml.CNOT(0, 1)
                qml.RY(weights[layer, 0], wires=0)
                qml.RY(weights[layer, 1], wires=1)
                qml.CNOT(0, 1)
                qml.RY(weights[layer, 2], wires=0)
                qml.RY(weights[layer, 3], wires=1)
            # Return probabilities of qubit 0 (two outcomes)
            return qml.probs([0])
        self.circuit = circuit

    def sample(self, inputs: np.ndarray | None = None) -> np.ndarray:
        """Return sampled probabilities from the circuit."""
        if inputs is None:
            inputs = np.random.uniform(0, 2*np.pi, (2,))
        probs = self.circuit(inputs, self.weight_params)
        return probs

    def fit(self, data_loader, epochs: int = 200, lr: float = 0.01):
        """Train the variational sampler to match target distributions."""
        opt = AdamOptimizer(lr)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in data_loader:
                inputs = np.array(inputs)
                targets = np.array(targets)
                def loss_fn(params):
                    probs = self.circuit(inputs, params)
                    return np.sum(targets * np.log(probs + 1e-8))
                grads = opt.grad(loss_fn, self.weight_params)
                self.weight_params = opt.apply_gradients(self.weight_params, grads)
                epoch_loss += loss_fn(self.weight_params)
            # print(f"Epoch {epoch+1}/{epochs} loss: {epoch_loss/len(data_loader):.4f}")

__all__ = ["SamplerQNN"]
