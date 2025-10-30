"""Hybrid estimator combining quantum FastBaseEstimator, photonic fraud detection, and QCNN circuits."""
from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence, Callable
from dataclasses import dataclass

def _ensure_batch(values: Sequence[float]) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes, params: FraudLayerParameters, clip: bool):
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

def build_photonic_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
):
    import strawberryfields as sf
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def build_qcnn() -> QuantumCircuit:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    import numpy as np

    def conv_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi/2, 1)
        target.cx(1,0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0,1)
        target.ry(params[2], 1)
        target.cx(1,0)
        target.rz(np.pi/2, 0)
        return target

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(conv_circuit(params[param_index:param_index+3]), [q1,q2])
            qc.barrier()
            param_index +=3
        for q1, q2 in zip(qubits[1::2], qubits[2::2]+[0]):
            qc = qc.compose(conv_circuit(params[param_index:param_index+3]), [q1,q2])
            qc.barrier()
            param_index +=3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi/2, 1)
        target.cx(1,0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0,1)
        target.ry(params[2], 1)
        return target

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources)+len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits//2 * 3)
        for source,sink in zip(sources,sinks):
            qc = qc.compose(pool_circuit(params[param_index:param_index+3]), [source,sink])
            qc.barrier()
            param_index +=3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc

    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(conv_layer(8,"c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0,1,2,3],[4,5,6,7],"p1"), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4,"c2"), list(range(4,8)), inplace=True)
    ansatz.compose(pool_layer([0,1],[2,3],"p2"), list(range(4,8)), inplace=True)
    ansatz.compose(conv_layer(2,"c3"), list(range(6,8)), inplace=True)
    ansatz.compose(pool_layer([0],[1],"p3"), list(range(6,8)), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(ansatz, list(range(8)), inplace=True)
    return circuit

class FastHybridEstimator:
    """Evaluate a Qiskit circuit for batches of parameter sets and observables, with optional shot‑noise simulation."""
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
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                rng.normal(loc=val.real, scale=1/np.sqrt(shots)) + 1j * rng.normal(loc=val.imag, scale=1/np.sqrt(shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FastHybridEstimator", "FraudLayerParameters", "build_photonic_fraud_detection_program", "build_qcnn"]
