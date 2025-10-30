"""Hybrid quantum estimator that builds a QCNN ansatz and can be combined
with a classical PyTorch model for hybrid experiments.  The estimator
leverages Qiskit’s EstimatorQNN and supports shot noise via the
underlying FastBaseEstimator interface.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Optional

import torch
from torch import nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# ---- Quantum QCNN circuit construction ----
def conv_circuit(params: Sequence[float]) -> QuantumCircuit:
    """Two‑qubit convolution unitary used in the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def pool_circuit(params: Sequence[float]) -> QuantumCircuit:
    """Two‑qubit pooling unitary used in the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Builds a convolutional layer acting on adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = conv_circuit(params[3 * (q1 // 2) : 3 * (q1 // 2 + 1)])
        qc.append(sub, [q1, q2])
        qc.barrier()
    return qc

def pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
    """Builds a pooling layer that maps source qubits to sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, sink, idx in zip(sources, sinks, range(len(sources))):
        sub = pool_circuit(params[3 * idx : 3 * (idx + 1)])
        qc.append(sub, [src, sink])
        qc.barrier()
    return qc

def QCNN() -> EstimatorQNN:
    """Constructs the full QCNN ansatz and returns an EstimatorQNN."""
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First convolution + pooling
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)

    # Second convolution + pooling
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)

    # Third convolution + pooling
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

# ---- Base quantum estimator ----
class FastBaseEstimator:
    """Evaluates expectation values of a parametrized quantum circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[SparsePauliOp], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# ---- Hybrid quantum estimator ----
class FastHybridEstimator(FastBaseEstimator):
    """Hybrid estimator that wraps a QCNN ansatz and can optionally
    attach a classical PyTorch model for hybrid experiments."""
    def __init__(self, *, qnn: EstimatorQNN, classical_model: Optional[nn.Module] = None) -> None:
        super().__init__(qnn.circuit)
        self.qnn = qnn
        self.classical_model = classical_model

    def evaluate_quantum(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        return self.qnn.evaluate(observables, parameter_sets)

    def evaluate_classical(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        if self.classical_model is None:
            raise RuntimeError("No classical model attached.")
        # Reuse FastEstimator logic
        estimator = FastEstimator(self.classical_model)
        return estimator.evaluate(observables, parameter_sets)

    def attach_classical_model(self, model: nn.Module) -> None:
        self.classical_model = model

__all__ = ["QCNN", "FastBaseEstimator", "FastHybridEstimator"]
