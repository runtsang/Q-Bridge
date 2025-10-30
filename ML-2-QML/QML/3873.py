import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from typing import Iterable, Union

class HybridFCL(nn.Module):
    """
    Quantum‑enhanced fully‑connected layer built on PennyLane.
    In quantum mode a parameterised RX‑CNOT circuit produces expectation values
    that are then linearly mapped to the desired hidden dimension.
    Classical mode falls back to a linear + tanh to preserve backward compatibility.
    """
    def __init__(self, n_features: int = 1, hidden_dim: int = 1, n_qubits: int = 0, use_quantum: bool = False):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.use_quantum = use_quantum

        if use_quantum:
            if n_qubits <= 0:
                raise ValueError("n_qubits must be > 0 when use_quantum=True")
            self.n_qubits = n_qubits
            # Device for the quantum circuit
            self.dev = qml.device("default.qubit", wires=n_qubits)
            # Linear mapping from input features to qubit parameters
            self.input_to_qubits = nn.Linear(n_features, n_qubits, bias=False)

            @qml.qnode(self.dev, interface="torch")
            def circuit(x: torch.Tensor):
                # Apply RX rotations with parameters x
                for i in range(n_qubits):
                    qml.RX(x[i], wires=i)
                # Entangle qubits with a CNOT chain
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Return expectation values of PauliZ on each qubit
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            self.circuit = circuit
            self.out_linear = nn.Linear(n_qubits, hidden_dim)
        else:
            self.linear = nn.Linear(n_features, hidden_dim)
            self.act = nn.Tanh()

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. ``thetas`` should be a tensor of shape (batch, n_features).
        """
        if self.use_quantum:
            # Map features to qubit parameters
            x = self.input_to_qubits(thetas)  # (batch, n_qubits)
            # Run the quantum circuit for each sample
            expectations = torch.stack([self.circuit(sample) for sample in x])
            # (batch, n_qubits)
            out = self.out_linear(expectations)
            return out
        else:
            out = self.linear(thetas)
            return self.act(out)

    def run(self, thetas: Union[Iterable[float], np.ndarray]) -> np.ndarray:
        """
        Convenience wrapper matching the original FCL API.
        Accepts an iterable or numpy array of input values and returns a numpy array.
        """
        if isinstance(thetas, (list, tuple)):
            thetas = np.array(thetas, dtype=np.float32)
        if isinstance(thetas, np.ndarray):
            thetas = torch.as_tensor(thetas, dtype=torch.float32)
        if thetas.ndim == 1:
            thetas = thetas.unsqueeze(0)
        output = self.forward(thetas)
        return output.detach().numpy()

__all__ = ["HybridFCL"]
