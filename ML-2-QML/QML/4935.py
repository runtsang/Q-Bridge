"""
HybridQCNet – quantum implementation that replaces each classical block with
a variational circuit or a Qiskit EstimatorQNN.  The public API matches the
classical version so both can be used interchangeably.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

# --------------------------------------------------------------------------- #
#  Quantum helper blocks – extracted and extended from the seed
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Variational self‑attention block implemented with Qiskit."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure_all()
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = backend.run(circuit, shots=shots)
        return job.result().get_counts(circuit)


class QuanvCircuit:
    """Quantum filter that mimics a quanvolution layer."""

    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend or AerSimulator()
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


class EstimatorQNNWrapper:
    """Thin wrapper around Qiskit’s EstimatorQNN that accepts numpy inputs."""

    def __init__(self):
        params = [Parameter("theta"), Parameter("phi")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)

        observable = SparsePauliOp.from_list([("Y", 1)])

        estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[params[0]],
            weight_params=[params[1]],
            estimator=estimator,
        )

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Return expectation values for a 1‑D array of inputs."""
        results = []
        for val in inputs:
            res = self.estimator_qnn.predict({ "theta": val })
            results.append(res)
        return np.array(results)


# --------------------------------------------------------------------------- #
#  HybridQCNetQuantum – quantum version of the hybrid network
# --------------------------------------------------------------------------- #
class HybridQCNetQuantum(nn.Module):
    """
    Quantum‑centric hybrid network that mirrors the classical HybridQCNet.
    The CNN backbone is replaced by quantum self‑attention and quanvolution
    blocks, and the final head is an EstimatorQNN.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backend = AerSimulator()
        self.self_attention = QuantumSelfAttention(n_qubits=4)
        self.quanv = QuanvCircuit(kernel_size=2, backend=self.backend, shots=100, threshold=0.5)
        self.estimator = EstimatorQNNWrapper()
        self.shift = np.pi / 2  # used for parameter shift in gradient (placeholder)

    def _counts_to_expectation(self, counts: dict) -> float:
        total = sum(counts.values())
        exp = 0.0
        for state, c in counts.items():
            z = 1 if state == "0" * len(state) else -1
            exp += z * c
        return exp / total

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: [batch, channels, H, W]
        batch = x.shape[0]
        # Flatten images to a vector
        flat = x.view(batch, -1)

        # 1. Quantum self‑attention
        attn_vals = []
        for i in range(batch):
            rotation = np.random.rand(12)
            entangle = np.random.rand(3)
            counts = self.self_attention.run(
                self.backend, rotation, entangle, shots=1024
            )
            attn_vals.append(self._counts_to_expectation(counts))
        attn_arr = np.array(attn_vals)

        # 2. Quanvolution on the first 4×4 patch
        patch = flat[:, :16].reshape(batch, 4, 4)
        quanv_vals = []
        for i in range(batch):
            quanv_vals.append(self.quanv.run(patch[i].numpy()))
        quanv_arr = np.array(quanv_vals)

        # 3. Combine signals (simple sum) and feed to EstimatorQNN
        combined = attn_arr + quanv_arr
        probs = self.estimator.run(combined)

        probs_tensor = torch.tensor(probs, dtype=torch.float32, device=x.device)
        return torch.cat((probs_tensor, 1 - probs_tensor), dim=-1)


__all__ = ["HybridQCNetQuantum"]
