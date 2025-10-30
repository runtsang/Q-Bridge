"""QuanvolutionHybrid – quantum side (anchor: Quanvolution.py)."""

from __future__ import annotations

import torchquantum as tq
import torch
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
from typing import Iterable, List, Tuple

# --------------------------------------------------------------------------- #
# Quantum quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(tq.QuantumModule):
    """Random two‑qubit kernel applied to 2×2 image patches."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches: List[torch.Tensor] = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))

        return torch.cat(patches, dim=1)

# --------------------------------------------------------------------------- #
# Quantum self‑attention block
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Self‑attention style circuit built with Qiskit."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend: qiskit.providers.basebackend.BaseBackend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

# --------------------------------------------------------------------------- #
# EstimatorQNN using qiskit
# --------------------------------------------------------------------------- #
def EstimatorQNN() -> QiskitEstimatorQNN:
    """Return a simple Qiskit EstimatorQNN configured for a single‑qubit circuit."""
    # Parameterised circuit
    input1 = qiskit.circuit.Parameter("input1")
    weight1 = qiskit.circuit.Parameter("weight1")
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(input1, 0)
    qc.rx(weight1, 0)

    # Observable
    observable = SparsePauliOp.from_list([("Y", 1)])

    # Estimator
    estimator = StatevectorEstimator()

    # Qiskit EstimatorQNN wrapper
    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[input1],
        weight_params=[weight1],
        estimator=estimator,
    )

__all__ = [
    "QuanvolutionFilter",
    "QuantumSelfAttention",
    "EstimatorQNN",
]
