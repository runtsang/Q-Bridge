"""
Hybrid quantum‑classical classifier that uses a variational quantum circuit
as the final head.  The module mirrors the classical counterpart and
provides a unified API.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Sequence, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit import ParameterVector

# --------------------------------------------------------------------------- #
#  Quantum circuit wrapper – executes a parametrised circuit
# --------------------------------------------------------------------------- #
class QuantumCircuitWrapper:
    """
    Wrapper around a parametrised Qiskit circuit that can be executed on
    a chosen backend with a fixed number of shots.  The ``run`` method
    accepts a list of parameter sets and returns expectation values
    for the observables defined in ``build_classifier_circuit``.
    """
    def __init__(
        self,
        circuit: QuantumCircuit,
        observables: List[SparsePauliOp],
        backend: qiskit.providers.Backend | None = None,
        shots: int = 1024,
    ):
        self.circuit = circuit
        self.observables = observables
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.parameters = list(circuit.parameters)

    def run(self, parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Execute the circuit for each set of parameters and return expectations.
        The implementation uses Statevector for deterministic expectation values.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self.circuit.assign_parameters(dict(zip(self.parameters, params)), inplace=False)
            state = Statevector.from_instruction(bound)
            expectations = [state.expectation_value(obs) for obs in self.observables]
            results.append(expectations)
        return results

# --------------------------------------------------------------------------- #
#  Build quantum classifier circuit
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered variational ansatz with explicit encoding and
    variational parameters.  The returned tuple mirrors the classical
    version:
        - circuit : QuantumCircuit
        - encoding : list of ParameterVector objects for data encoding
        - weights : list of ParameterVector objects for variational parameters
        - observables : list of SparsePauliOp observables to measure
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Data encoding
    for idx, qubit in enumerate(range(num_qubits)):
        circuit.rx(encoding[idx], qubit)

    # Variational layers
    w_idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[w_idx], qubit)
            w_idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables – Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return circuit, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
#  Hybrid estimator for quantum circuits
# --------------------------------------------------------------------------- #
class HybridEstimator:
    """
    Evaluator that can handle a QuantumCircuitWrapper.  It mimics the
    interface of the classical estimator but works directly with the
    quantum backend.
    """
    def __init__(self, circuit: QuantumCircuitWrapper):
        self.circuit = circuit

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        For each parameter set the quantum circuit is executed and the
        expectation values are returned.  The ``observables`` argument
        is kept for API compatibility and is ignored internally.
        """
        raw = self.circuit.run(parameter_sets)
        # Convert complex to real if needed
        results: List[List[float]] = [[float(val.real) if hasattr(val, "real") else float(val) for val in row] for row in raw]
        return results

# --------------------------------------------------------------------------- #
#  Unified classifier – classical backbone + quantum head
# --------------------------------------------------------------------------- #
class UnifiedClassifier(nn.Module):
    """
    Hybrid model that uses a classical feed‑forward backbone to produce
    parameters for a variational quantum circuit.  The final output is
    obtained by measuring Z on each qubit.
    """
    def __init__(
        self,
        num_features: int,
        depth: int,
        *,
        num_qubits: int | None = None,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 1024,
    ):
        super().__init__()
        # Classical backbone
        self.backbone, self.encoding, self.weight_sizes, _ = build_classifier_circuit(num_features, depth)
        # Quantum head
        if num_qubits is None:
            num_qubits = max(2, num_features)  # fallback
        circuit, _, _, observables = build_classifier_circuit(num_qubits, depth)
        if backend is None:
            backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_head = QuantumCircuitWrapper(circuit, observables, backend, shots)
        self.observables = observables
        self.estimator = HybridEstimator(self.quantum_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through the backbone to obtain a feature vector that
        is then interpreted as a set of parameters for the quantum circuit.
        The quantum circuit is executed and the expectation values are
        returned as a probability vector.
        """
        features = self.backbone(x).squeeze()
        # Ensure features shape matches number of parameters expected by the circuit
        param_set = features.tolist()
        expectations = self.quantum_head.run([param_set])[0]
        probs = torch.tensor(expectations, dtype=torch.float32)
        return probs

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Delegate to the quantum estimator.  The ``observables`` argument
        is accepted for API compatibility but ignored.
        """
        return self.estimator.evaluate(observables, parameter_sets)

__all__ = ["QuantumCircuitWrapper", "build_classifier_circuit", "HybridEstimator", "UnifiedClassifier"]
