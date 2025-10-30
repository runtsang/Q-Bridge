"""Unified FastBaseEstimator for quantum circuits with shot‑noise support and helper factories.

The quantum module mirrors the classical API but operates on
parameterised ``qiskit.QuantumCircuit`` objects.  It can evaluate
state‑vector expectations or run a shot‑based simulation, and ships
factory helpers for a QCNN ansatz, a simple fully‑connected layer,
and a hybrid classifier that uses a quantum expectation head.
"""
from __future__ import annotations

import numpy as np
from typing import Iterable, Sequence, List

from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Estimator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parameterised quantum circuit."""
    def __init__(self, circuit: QuantumCircuit, backend: Aer.AerSimulator | None = None, shots: int | None = None):
        self._circuit = circuit
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        self.estimator = Estimator()

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with the supplied parameter values bound."""
        if len(parameter_values)!= len(self._circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._circuit.parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        If *shots* is ``None`` the state‑vector simulation is used, otherwise
        the circuit is executed on the Aer simulator with the requested
        number of shots.  The optional ``seed`` argument controls the
        simulator RNG.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        shots = shots if shots is not None else self.shots
        for values in parameter_sets:
            bound = self._bind(values)
            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = self.estimator.run(bound, observables, shots=shots, seed_simulator=seed)
                result = job.result()
                row = [res.values[0] for res in result]
            results.append(row)
        return results


def QCNN() -> QuantumCircuit:
    """Return a parameterised 8‑qubit QCNN ansatz."""
    def conv_circuit(params):
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

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = conv_circuit(params[param_index:param_index + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            sub = conv_circuit(params[param_index:param_index + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            sub = pool_circuit(params[param_index:param_index + 3])
            qc.append(sub, [source, sink])
            qc.barrier()
            param_index += 3
        return qc

    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)
    return circuit


def FCL() -> QuantumCircuit:
    """Return a simple 1‑qubit parameterised circuit mimicking a fully‑connected layer."""
    qc = QuantumCircuit(1)
    theta = ParameterVector("theta", 1)
    qc.h(0)
    qc.barrier()
    qc.ry(theta[0], 0)
    qc.measure_all()
    return qc


class QCNet:
    """Hybrid classifier built on the QCNN ansatz with a quantum expectation head."""
    def __init__(self, backend=Aer.get_backend("aer_simulator"), shots: int = 1024):
        self.circuit = QCNN()
        self.backend = backend
        self.shots = shots
        self.estimator = Estimator()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Run the network for a batch of input feature vectors (shape: N x 8)."""
        probs = []
        all_params = list(self.circuit.parameters)
        feature_params = all_params[:8]
        ansatz_params = all_params[8:]
        for vec in inputs:
            param_values = list(vec) + [0.0] * len(ansatz_params)
            bound = self.circuit.assign_parameters(dict(zip(all_params, param_values)), inplace=False)
            result = self.estimator.run(bound, [self.observable], shots=self.shots).result()
            exp_val = result.values[0]
            prob = 1 / (1 + np.exp(-exp_val))  # sigmoid
            probs.append(prob)
        return np.array(probs)


__all__ = [
    "FastBaseEstimator",
    "QCNN",
    "FCL",
    "QCNet",
]
