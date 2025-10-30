"""Hybrid quantum estimator with QCNN and sampler support.

The :class:`HybridEstimator` class wraps a parameterised QuantumCircuit
and evaluates expectation values or sampled estimates.  Two convenience
constructors, :func:`QCNN` and :func:`SamplerQNN`, build the full QCNN ansatz
and a simple soft‑max sampler circuit, mirroring the classical models.
The design uses Qiskit’s StatevectorEstimator for deterministic expectations
and StatevectorSampler for shot‑limited sampling.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Sequence as Seq

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.circuit.library import ZFeatureMap


class HybridEstimator:
    """Evaluate a parameterised quantum circuit with optional shot noise.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to be evaluated.  It may contain both input and weight parameters.
    estimator : StatevectorEstimator | StatevectorSampler | None
        Backend used for evaluation.  If None, a deterministic
        StatevectorEstimator is instantiated.  For sampling tasks
        a StatevectorSampler should be passed explicitly.
    input_params : Sequence[ParameterVector] | None
        Parameters that encode the data (used when estimator is EstimatorQNN).
    weight_params : Sequence[ParameterVector] | None
        Trainable parameters of the ansatz.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        estimator: StatevectorEstimator | StatevectorSampler | None = None,
        input_params: Sequence[ParameterVector] | None = None,
        weight_params: Sequence[ParameterVector] | None = None,
    ) -> None:
        self.circuit = circuit
        self.estimator = estimator or StatevectorEstimator()
        self.input_params = list(input_params) if input_params else []
        self.weight_params = list(weight_params) if weight_params else []

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with the given parameters bound."""
        all_params = self.input_params + self.weight_params
        if len(param_values)!= len(all_params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(all_params, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return expectation values or sampled estimates.

        If ``shots`` is provided, the estimator is switched to a noisy
        sampler backend; otherwise the deterministic StatevectorEstimator
        is used.  The method is agnostic to whether the circuit is a
        QCNN ansatz or a sampler circuit.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for params in parameter_sets:
            bound = self._bind(params)

            if shots is None:
                # Deterministic expectation values
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Sample-based estimates
                sampler = StatevectorSampler(seed=seed)
                sample_counts = sampler.run(bound, shots=shots).result().get_counts()
                row = []
                for obs in observables:
                    if isinstance(obs, SparsePauliOp):
                        # Assume single‑qubit Z for simplicity
                        exp = 0.0
                        for bitstring, count in sample_counts.items():
                            # Qiskit uses little‑endian bit ordering
                            bit = int(bitstring[0])
                            exp += (1 if bit == 0 else -1) * count
                        exp /= shots
                        row.append(complex(exp))
                    else:
                        # Fallback to deterministic evaluation
                        state = Statevector.from_instruction(bound)
                        row.append(state.expectation_value(obs))
            results.append(row)

        return results


# --------------------------------------------------------------------------- #
# QCNN helper
# --------------------------------------------------------------------------- #
def QCNN() -> HybridEstimator:
    """Return a HybridEstimator wrapping the full QCNN ansatz.

    The circuit consists of a ZFeatureMap feature map followed by
    three convolution‑and‑pooling stages, exactly as in the
    QML seed.  The estimator used is a StatevectorEstimator to
    provide deterministic expectation values; for noisy runs
    the caller can pass ``shots`` to the evaluate method.
    """
    # Feature map
    feature_map = ZFeatureMap(8)

    # Convolution helper
    def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

    def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolution")
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            slice_params = params[i // 2 * 3 : i // 2 * 3 + 3]
            qc.compose(conv_circuit(slice_params), [i, i + 1], inplace=True)
        return qc

    def pool_circuit(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer(sources: Sequence[int], sinks: Sequence[int], prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling")
        params = ParameterVector(prefix, length=len(sources) * 3)
        for idx, (s, t) in enumerate(zip(sources, sinks)):
            slice_params = params[idx * 3 : idx * 3 + 3]
            qc.compose(pool_circuit(slice_params), [s, t], inplace=True)
        return qc

    # Build ansatz
    ansatz = QuantumCircuit(8)
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Full circuit: feature map + ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    return HybridEstimator(
        circuit=circuit,
        estimator=StatevectorEstimator(),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )


# --------------------------------------------------------------------------- #
# Sampler helper
# --------------------------------------------------------------------------- #
def SamplerQNN() -> HybridEstimator:
    """Return a HybridEstimator wrapping the sampler circuit.

    The circuit implements a two‑qubit circuit with Ry rotations for
    inputs and trainable Ry rotations plus a CX.  The estimator is
    a StatevectorSampler to emulate measurement statistics.
    """
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    return HybridEstimator(
        circuit=qc,
        estimator=StatevectorSampler(),
        input_params=inputs,
        weight_params=weights,
    )


__all__ = ["HybridEstimator", "QCNN", "SamplerQNN"]
