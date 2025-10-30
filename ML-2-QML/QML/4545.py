"""Hybrid quantum model combining a variational circuit, an encoder,
a Qiskit‑based self‑attention block, and a regression head.

The `HybridNAT` class inherits from `torchquantum.QuantumModule`
and mirrors the classical version in terms of interfaces while
leveraging quantum expressivity.  It includes a `FastEstimator`
wrapper that evaluates expectation values of Pauli‑Z observables for
parameter sets, inspired by the lightweight FastEstimator in the
reference pair.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# --- Quantum self‑attention helper --------------------------------------------
class QuantumSelfAttention:
    """Self‑attention style block implemented with Qiskit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
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
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

backend = qiskit.Aer.get_backend("qasm_simulator")
q_attention = QuantumSelfAttention(n_qubits=4)

# --- FastEstimator for quantum circuits --------------------------------------
class FastQuantumEstimator:
    """
    Lightweight estimator that evaluates a `torchquantum.QuantumModule`
    for a set of parameter vectors and a list of Pauli‑Z observables.
    """
    def __init__(self, circuit: tq.QuantumModule):
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, values: list[float]) -> tq.QuantumModule:
        if len(values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.parameters, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: list[tq.PauliZ],
        parameter_sets: list[list[float]],
    ) -> list[list[complex]]:
        results = []
        for params in parameter_sets:
            bound_circuit = self._bind(params)
            state = tq.StateVector.from_instruction(bound_circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# --- Hybrid quantum model ------------------------------------------------------
class HybridNAT(tq.QuantumModule):
    """
    Quantum hybrid model that encodes a 4‑qubit state, applies a variational
    layer, a self‑attention style Qiskit circuit, measures Pauli‑Z, and
    projects to a 4‑dimensional output.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 4)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Complex state vector of shape (batch, 2**n_wires).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch, 4).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        measured = self.measure(qdev)  # (bsz, n_wires)
        out = self.head(measured)
        return out

    def estimator(self) -> FastQuantumEstimator:
        """Return a FastQuantumEstimator that evaluates Pauli‑Z observables."""
        return FastQuantumEstimator(self)

__all__ = ["HybridNAT"]
