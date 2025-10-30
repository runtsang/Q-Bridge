"""Hybrid estimator combining quantum Qiskit circuits with flexible observables and shot noise.

This module defines `HybridFastEstimator`, a lightweight wrapper around any
`qiskit.QuantumCircuit`.  The estimator accepts batches of parameter vectors,
binds them to the circuit, and evaluates a list of `BaseOperator`
observables.  An optional subclass `FastEstimator` adds Gaussian shot noise,
mimicking the behaviour of a noisy quantum device.

Factory helpers return the quantum counterparts of the classical seeds:
- `QFCModel` (torchquantum implementation)
- `QCNN` (EstimatorQNN)
- `FCL` (simple parameterised Qiskit circuit)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
import torchquantum as tq
import torchquantum.functional as tqf
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit.circuit import ParameterVector

ScalarObservable = BaseOperator


def _ensure_batch(values: Sequence[float]) -> List[float]:
    """Return a list of parameter values."""
    return list(values)


class HybridFastEstimator:
    """Evaluate a QuantumCircuit for a list of parameter sets."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class FastEstimator(HybridFastEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
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
            noisy_row = [rng.normal(float(mean), max(1e-6, 1 / shots)) for mean in row]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# Factory helpers mirroring the quantum seeds
# --------------------------------------------------------------------------- #

def QFCModel() -> tq.QuantumModule:
    """Return the quantum QFCModel (torchquantum implementation)."""

    class QFCModel(tq.QuantumModule):
        class QLayer(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = 4
                self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
                self.rx0 = tq.RX(has_params=True, trainable=True)
                self.ry0 = tq.RY(has_params=True, trainable=True)
                self.rz0 = tq.RZ(has_params=True, trainable=True)
                self.crx0 = tq.CRX(has_params=True, trainable=True)

            @tq.static_support
            def forward(self, qdev: tq.QuantumDevice):
                self.random_layer(qdev)
                self.rx0(qdev, wires=0)
                self.ry0(qdev, wires=1)
                self.rz0(qdev, wires=3)
                self.crx0(qdev, wires=[0, 2])
                tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
                tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
                tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
            self.q_layer = self.QLayer()
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.norm = nn.BatchNorm1d(self.n_wires)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
            pooled = F.avg_pool2d(x, 6).view(bsz, 16)
            self.encoder(qdev, pooled)
            self.q_layer(qdev)
            out = self.measure(qdev)
            return self.norm(out)

    return QFCModel()


def QCNN() -> tq.QuantumModule:
    """Return the quantum QCNN model implemented with EstimatorQNN."""
    from qiskit import Aer
    from qiskit.quantum_info.operators.base_operator import BaseOperator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
    from qiskit_machine_learning.optimizers import COBYLA
    from qiskit_machine_learning.utils import algorithm_globals
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import ZFeatureMap
    from qiskit.primitives import StatevectorEstimator as Estimator

    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    def conv_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    params = ParameterVector("θ", length=3)
    circuit = conv_circuit(params)

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    circuit = conv_layer(4, "θ")

    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    params = ParameterVector("θ", length=3)
    circuit = pool_circuit(params)

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc

    sources = [0, 1]
    sinks = [2, 3]
    circuit = pool_layer(sources, sinks, "θ")

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

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn


def FCL() -> QuantumCircuit:
    """Return a simple parameterised quantum circuit for a fully connected layer."""
    class QuantumCircuitWrapper:
        def __init__(self, n_qubits: int, backend, shots: int):
            self._circuit = QuantumCircuit(n_qubits)
            self.theta = ParameterVector("theta", length=1)
            self._circuit.h(range(n_qubits))
            self._circuit.barrier()
            self._circuit.ry(self.theta, range(n_qubits))
            self._circuit.measure_all()
            self.backend = backend
            self.shots = shots

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            job = self.backend.run(
                self._circuit,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
            result = job.result()
            counts = np.array(list(result.get_counts(self._circuit).values()))
            states = np.array(list(result.get_counts(self._circuit).keys())).astype(float)
            probabilities = counts / self.shots
            expectation = np.sum(states * probabilities)
            return np.array([expectation])

    simulator = Aer.get_backend("qasm_simulator")
    circuit = QuantumCircuitWrapper(1, simulator, 100)
    return circuit


__all__ = [
    "HybridFastEstimator",
    "FastEstimator",
    "QFCModel",
    "QCNN",
    "FCL",
]
