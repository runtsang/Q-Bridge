"""Hybrid quantum regression model that encodes data into a quantum circuit,
applies a parameterised quantum layer, measures expectation values, and
uses a quantum sampler to compute an additional probabilistic output.
The model is a drop‑in replacement for the classical HybridRegression
while leveraging qiskit and torchquantum for quantum‑centric operations.

The implementation intentionally keeps the quantum and classical parts
independent so the same class name can be used in both modules.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
try:
    # qiskit imports are optional – the module will still import
    import qiskit
    from qiskit import QuantumCircuit, Aer
    from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
    from qiskit.primitives import StatevectorSampler
except Exception:
    qiskit = None  # pragma: no cover
    QiskitSamplerQNN = None
    StatevectorSampler = None

from torch.utils.data import Dataset
from typing import Iterable, Tuple, Sequence

# ----------------------------------------------------------------------
# Dataset and data generation – identical to the classical counterpart
# ----------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data in the form of complex state vectors."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset(Dataset):
    """PyTorch dataset for quantum state vectors and labels."""
    def __init__(self, samples: int = 1024, num_wires: int = 3):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"states": torch.tensor(self.states[idx], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

# ----------------------------------------------------------------------
# Quantum network components
# ----------------------------------------------------------------------
class QLayer(tq.QuantumModule):
    """Parameterized quantum layer with random gates and single‑qubit rotations."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)

class HybridRegression(tq.QuantumModule):
    """Quantum regression model that mirrors the classical HybridRegression
    but replaces feature extraction with a quantum encoder and uses a
    qiskit sampler for an additional expectation value.
    """
    def __init__(self, num_wires: int = 3):
        super().__init__()
        self.n_wires = num_wires
        # 1. Quantum encoder – simple Ry rotation per qubit
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # 2. Parameterised quantum layer
        self.q_layer = QLayer(num_wires)
        # 3. Measurement – Pauli‑Z expectation on all qubits
        self.measure = tq.MeasureAll(tq.PauliZ)
        # 4. Classical head
        self.head = nn.Linear(num_wires, 1)
        # 5. Optional qiskit sampler circuit
        if qiskit is not None:
            self._setup_sampler()
        else:
            self.sampler = None

    # ------------------------------------------------------------------
    # Qiskit sampler – small circuit that returns a probability distribution
    # ------------------------------------------------------------------
    def _setup_sampler(self):
        """Create a 1‑qubit sampler that measures the Z expectation."""
        circ = QuantumCircuit(1)
        theta = qiskit.circuit.Parameter("theta")
        circ.ry(theta, 0)
        circ.measure_all()
        backend = Aer.get_backend("qasm_simulator")
        self.sampler = StatevectorSampler(backend)

    def _sampler_expectation(self, angle: float) -> float:
        """Run the qiskit sampler for a single rotation angle."""
        if self.sampler is None:
            return 0.0
        circ = QuantumCircuit(1)
        theta = qiskit.circuit.Parameter("theta")
        circ.ry(theta, 0)
        circ.measure_all()
        job = self.sampler.run(circ, parameter_binds=[{theta: angle}])
        result = job.result()
        counts = result.get_counts(circ)
        probs = np.array(list(counts.values())) / sum(counts.values())
        outcomes = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        return float(np.dot(outcomes, probs))  # expectation of |1>

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Complex state vectors of shape (N, 2**n_wires).

        Returns
        -------
        torch.Tensor
            Regression target as the mean of a classical head and a quantum
            sampler expectation.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        # Encode raw states
        self.encoder(qdev, state_batch)
        # Apply parameterised layer
        self.q_layer(qdev)
        # Measure to obtain a feature vector
        features = self.measure(qdev)  # shape (N, n_wires)
        # Classical head
        head_out = self.head(features).squeeze(-1)  # shape (N,)

        # Quantum sampler part – use head_out as rotation angles
        if self.sampler is not None:
            sampler_vals = []
            for angle in head_out.detach().cpu().numpy():
                sampler_vals.append(self._sampler_expectation(float(angle)))
            sampler_out = torch.tensor(sampler_vals, dtype=torch.float32, device=head_out.device)
            # Combine
            output = 0.5 * head_out + 0.5 * sampler_out
        else:
            output = head_out

        return output

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
