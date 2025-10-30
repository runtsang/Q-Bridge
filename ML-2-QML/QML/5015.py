"""
Hybrid quantum regression module that combines torchquantum
variational circuits with a Qiskit‑based self‑attention block
and a classical linear head.  It mirrors the structure of the
original QuantumRegression example while adding richer
quantum features.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info import Statevector, Pauli
from typing import Sequence, Iterable, List, Callable

# ----------------------------------------------------
# Data generation
# ----------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    The labels are a non‑linear function of the angles.
    """
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
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset returning quantum states and scalar targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ----------------------------------------------------
# Quantum attention block (Qiskit)
# ----------------------------------------------------
class QuantumAttentionCircuit:
    """
    A lightweight self‑attention style circuit built with Qiskit.
    The circuit prepares an entangled state, applies a simple rotation
    pattern and measures in the computational basis.  The expectation
    values of Pauli‑Z on each qubit are used as additional features.
    """
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.base_circuit = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self.base_circuit.ry(np.pi / 4, i)
        for i in range(n_qubits - 1):
            self.base_circuit.cx(i, i + 1)

    def run(self, state: np.ndarray) -> np.ndarray:
        """
        Evaluate the attention circuit on a given state vector.
        Returns an array of Pauli‑Z expectation values per qubit.
        """
        circuit = QuantumCircuit(self.n_qubits)
        circuit.append(Statevector(state).to_instruction(), range(self.n_qubits))
        circuit.append(self.base_circuit, range(self.n_qubits))
        job = execute(circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # Compute expectation value of Z for each qubit
        exp = np.zeros(self.n_qubits)
        total = sum(counts.values())
        for bitstring, cnt in counts.items():
            prob = cnt / total
            for i, bit in enumerate(reversed(bitstring)):
                exp[i] += prob * (1 if bit == "0" else -1)
        return exp

# ----------------------------------------------------
# Estimator utilities
# ----------------------------------------------------
class FastBaseEstimatorQuantum:
    """
    Evaluate expectation values of Pauli operators for a parametrised
    Qiskit circuit.  Mimics the lightweight estimator in the original
    QML repo.
    """
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Pauli],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class FastEstimatorQuantum(FastBaseEstimatorQuantum):
    """
    Adds optional Gaussian shot noise to the deterministic estimator.
    """
    def evaluate(
        self,
        observables: Iterable[Pauli],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                rng.normal(float(mean), max(1e-6, 1 / shots)) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

# ----------------------------------------------------
# Quantum‑classical hybrid regression model
# ----------------------------------------------------
class QuantumHybridRegression(tq.QuantumModule):
    """
    Variational regression model that encodes a quantum state,
    applies a parameterised layer, measures a set of observables,
    and concatenates the result with a Qiskit attention feature
    before passing through a classical linear head.
    """
    def __init__(self, num_wires: int, attention_qubits: int = 4):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self._build_qlayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical head
        self.head = nn.Linear(num_wires + attention_qubits, 1)

        # Quantum attention block
        self.attention = QuantumAttentionCircuit(
            n_qubits=attention_qubits,
            shots=2048,
        )

    def _build_qlayer(self, n_wires: int) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.random = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
                self.rx = tq.RX(has_params=True, trainable=True)
                self.ry = tq.RY(has_params=True, trainable=True)

            def forward(self, qdev: tq.QuantumDevice):
                self.random(qdev)
                for w in range(self.n_wires):
                    self.rx(qdev, wires=w)
                    self.ry(qdev, wires=w)
        return QLayer(n_wires)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode input states, apply the variational layer, measure,
        compute quantum attention features, and return predictions.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        # Encode classical data
        self.encoder(qdev, state_batch)

        # Variational layer
        self.q_layer(qdev)

        # Measurements
        features = self.measure(qdev)  # shape (bsz, n_wires)

        # Attention features (deterministic, no gradients)
        attn_feats = []
        for i in range(bsz):
            state_vec = state_batch[i].cpu().numpy()
            attn = self.attention.run(state_vec)
            attn_feats.append(attn)
        attn_tensor = torch.tensor(attn_feats, dtype=torch.float32, device=state_batch.device)

        # Concatenate and head
        combined = torch.cat([features, attn_tensor], dim=-1)
        return self.head(combined).squeeze(-1)

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "QuantumAttentionCircuit",
    "FastBaseEstimatorQuantum",
    "FastEstimatorQuantum",
    "QuantumHybridRegression",
]
