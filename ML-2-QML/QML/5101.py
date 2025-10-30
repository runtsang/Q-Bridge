"""Quantum estimator that maps input features to measurement outcomes and
aggregates these results via a fidelity‑derived graph of the batch."""
from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple, Sequence

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

# --------------------------------------------------------------------------- #
# Fidelity utilities
# --------------------------------------------------------------------------- #
def _pairwise_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two state vectors."""
    return float(torch.abs(torch.dot(a, b)) ** 2)


def fidelity_adjacency(
    states: torch.Tensor,
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> torch.Tensor:
    """Return an adjacency matrix derived from pairwise fidelities."""
    n = states.shape[0]
    adj = torch.zeros((n, n), dtype=states.dtype, device=states.device)
    for i in range(n):
        for j in range(i + 1, n):
            fid = _pairwise_fidelity(states[i], states[j])
            if fid >= threshold:
                adj[i, j] = adj[j, i] = 1.0
            elif secondary is not None and fid >= secondary:
                adj[i, j] = adj[j, i] = secondary_weight
    return adj

# --------------------------------------------------------------------------- #
# Quantum estimator
# --------------------------------------------------------------------------- #
class EstimatorQNN:
    """Hybrid quantum estimator that maps input features to a statevector,
    measures all qubits, and aggregates measurement results via a
    fidelity‑based graph of the sample batch.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        seed: int | None = None,
        threshold: float = 0.8,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.threshold = threshold
        rng = np.random.default_rng(seed)
        # Create a template circuit with parametric rotation layers
        self.base_circuit = QuantumCircuit(n_qubits)
        self.params: List[Parameter] = []
        for l in range(n_layers):
            for q in range(n_qubits):
                p = Parameter(f"θ_{l}_{q}")
                self.params.append(p)
                self.base_circuit.ry(p, q)
            for q in range(n_qubits - 1):
                self.base_circuit.cx(q, q + 1)
        # Random initial values for the weight parameters
        self.param_values = rng.normal(size=len(self.params))
        self.backend = Aer.get_backend("statevector_simulator")

    def _encode_input(self, circuit: QuantumCircuit, inputs: torch.Tensor) -> QuantumCircuit:
        """Encode a feature vector into the first qubits using RY rotations."""
        for idx, val in enumerate(inputs):
            circuit.ry(val.item(), idx)
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute aggregated predictions for a batch of inputs."""
        batch = x.shape[0]
        outputs: List[torch.Tensor] = []
        for i in range(batch):
            circ = QuantumCircuit(self.n_qubits)
            circ = self._encode_input(circ, x[i])
            circ = circ.compose(self.base_circuit, front=False)
            circ = circ.bind_parameters(
                {p: v.item() for p, v in zip(self.params, self.param_values)}
            )
            job = execute(circ, self.backend)
            result = job.result()
            statevec = result.get_statevector(circ)
            # Compute expectation of Pauli‑Z on each qubit
            exp_z = []
            for q in range(self.n_qubits):
                exp = 0.0
                for idx, amp in enumerate(statevec):
                    bit = (idx >> (self.n_qubits - q - 1)) & 1
                    exp += np.abs(amp) ** 2 * ((-1) ** bit)
                exp_z.append(exp)
            outputs.append(torch.tensor(exp_z, dtype=torch.float32))
        outputs = torch.stack(outputs, dim=0)
        # Build adjacency matrix from pairwise fidelities
        adj = fidelity_adjacency(outputs, self.threshold)
        # Graph‑aggregated representation
        aggregated = torch.matmul(adj, outputs)
        return aggregated

    def generate_random_dataset(
        self, samples: int = 100
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Return synthetic (input, output) pairs and the adjacency graph
        for a batch of randomly sampled feature vectors.
        """
        rng = np.random.default_rng()
        inputs = torch.tensor(
            rng.normal(size=(samples, self.n_qubits)), dtype=torch.float32
        )
        outputs: List[torch.Tensor] = []
        for i in range(samples):
            circ = QuantumCircuit(self.n_qubits)
            circ = self._encode_input(circ, inputs[i])
            circ = circ.compose(self.base_circuit, front=False)
            circ = circ.bind_parameters(
                {p: v.item() for p, v in zip(self.params, self.param_values)}
            )
            job = execute(circ, self.backend)
            result = job.result()
            statevec = result.get_statevector(circ)
            exp_z = []
            for q in range(self.n_qubits):
                exp = 0.0
                for idx, amp in enumerate(statevec):
                    bit = (idx >> (self.n_qubits - q - 1)) & 1
                    exp += np.abs(amp) ** 2 * ((-1) ** bit)
                exp_z.append(exp)
            outputs.append(torch.tensor(exp_z, dtype=torch.float32))
        outputs = torch.stack(outputs, dim=0)
        adj = fidelity_adjacency(outputs, self.threshold)
        return [(inputs[i], outputs[i]) for i in range(samples)], adj

__all__ = ["EstimatorQNN"]
