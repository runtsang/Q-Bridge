from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator

class FastBaseEstimatorGen202:
    """Evaluate a Qiskit circuit for a batch of parameter sets with optional shot noise.

    Parameters
    ----------
    circuit
        A parametrized QuantumCircuit whose parameters will be bound to each
        parameter set passed to :meth:`evaluate`.
    """
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Return expectation values for each observable and parameter set.

        If ``shots`` is ``None`` a deterministic Statevector expectation is
        computed.  Otherwise the circuit is executed on the Aer simulator
        with sampling noise.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is None:
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
            return results

        simulator = AerSimulator(method="statevector", seed_simulator=seed)
        for values in parameter_sets:
            circ = self._bind(values)
            circ = transpile(circ, simulator)
            job = simulator.run(circ, shots=shots)
            result = job.result()
            counts = result.get_counts()
            row = []
            for obs in observables:
                exp = 0.0
                for bitstring, freq in counts.items():
                    # compute expectation of PauliZ product
                    val = 1
                    for qubit, char in enumerate(obs.to_label()):
                        if char == "Z":
                            val *= 1 if bitstring[-(qubit + 1)] == "0" else -1
                    exp += val * freq / shots
                row.append(exp)
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
#   Quantum ansatz builders inspired by QCNN and QFCModel                     #
# --------------------------------------------------------------------------- #

def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional layer used in the QCNN ansatz."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.append(_conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc.append(_conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
    """Pooling layer used in the QCNN ansatz."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.append(_pool_circuit(params[param_index : param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def qc_qcnn(num_qubits: int = 8) -> QuantumCircuit:
    """Return a QCNN ansatz circuit with convolution and pooling layers."""
    from qiskit.circuit.library import ZFeatureMap
    feature_map = ZFeatureMap(num_qubits)
    ansatz = QuantumCircuit(num_qubits)
    ansatz.compose(conv_layer(num_qubits, "c1"), list(range(num_qubits)), inplace=True)
    ansatz.compose(pool_layer(list(range(num_qubits)), list(range(num_qubits, 2 * num_qubits)), "p1"), list(range(num_qubits)), inplace=True)
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)
    return circuit

def qc_qfc() -> QuantumCircuit:
    """Return a simple 4‑qubit quantum circuit mimicking the QFCModel."""
    qc = QuantumCircuit(4)
    qc.h([0, 1, 2, 3])
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.rz(0.5, 0)
    qc.ry(0.3, 1)
    qc.rz(0.7, 2)
    qc.ry(0.2, 3)
    return qc

__all__ = ["FastBaseEstimatorGen202", "qc_qcnn", "qc_qfc"]
