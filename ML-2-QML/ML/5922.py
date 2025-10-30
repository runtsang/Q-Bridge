import torch
from torch import nn
import numpy as np
from typing import Iterable, Optional

class FCLHybrid(nn.Module):
    """
    Hybrid fully‑connected layer that supports both classical and quantum execution.
    * Classical mode: linear → Tanh → optional scaling/shift.
    * Quantum mode: delegates to a Qiskit circuit via ``run``.
    Parameters can be clipped and scaled as in the fraud‑detection example.
    """

    def __init__(self, n_features: int = 1, n_qubits: int = 1, use_quantum: bool = False):
        super().__init__()
        self.use_quantum = use_quantum
        self.n_features = n_features

        # Classical sub‑network
        self.linear = nn.Linear(n_features, 1)
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("shift", torch.zeros(1))

        # Placeholder for a quantum circuit
        self.quantum_circuit: Optional[object] = None

    def set_quantum_circuit(self, circuit: object) -> None:
        """Attach a Qiskit circuit that implements the same parameterisation."""
        self.quantum_circuit = circuit
        self.use_quantum = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass."""
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Quantum forward – returns expectation values from the attached Qiskit circuit.
        Raises an error if no circuit is attached.
        """
        if not self.use_quantum or self.quantum_circuit is None:
            raise RuntimeError("Quantum circuit not attached.")
        return self.quantum_circuit.run(thetas)
