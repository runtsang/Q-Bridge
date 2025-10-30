import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Pauli
from qiskit.providers.aer import StatevectorSimulator

class HybridEstimator(nn.Module):
    """
    Quantum feature extractor that maps a complex state vector to a set of
    expectation values.  For each sample we rotate each qubit by an angle
    derived from the corresponding component of the input state and then
    measure Pauli‑Z.  The resulting vector of expectation values is
    returned as a classical feature tensor.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        self.backend = Aer.get_backend("statevector_simulator")
        # Observable: Pauli‑Z on each qubit
        self.observable = Pauli("Z" * num_wires)

    def _build_circuit(self, angles: np.ndarray) -> QuantumCircuit:
        """
        Build a circuit that applies Ry(angles[i]) to qubit i.
        """
        qc = QuantumCircuit(self.num_wires)
        for i, angle in enumerate(angles):
            qc.ry(angle, i)
        return qc

    def _expectation(self, angles: np.ndarray) -> torch.Tensor:
        """
        Compute the expectation value of Pauli‑Z on all qubits for the given
        rotation angles.  Returns a tensor of shape (num_wires,).
        """
        qc = self._build_circuit(angles)
        job = execute(qc, self.backend, shots=1)
        result = job.result()
        statevec = result.get_statevector(qc)
        # Compute expectation of each Z
        exp_vals = []
        for i in range(self.num_wires):
            pauli_z = Pauli("Z" + "I" * (self.num_wires - i - 1))
            exp_vals.append(pauli_z.expectation_value(statevec).real)
        return torch.tensor(exp_vals, dtype=torch.float32)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        For each sample in the batch, take the first `num_wires` real parts
        of the input state as rotation angles, compute the quantum feature
        vector, and stack them into a tensor of shape (batch, num_wires).
        """
        batch_size = state_batch.shape[0]
        features = []
        for i in range(batch_size):
            angles = state_batch[i].real[:self.num_wires]
            features.append(self._expectation(angles.numpy()))
        return torch.stack(features, dim=0)

__all__ = ["HybridEstimator"]
