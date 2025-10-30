"""Hybrid quantum estimator built on Qiskit, with QCNN support."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional, Union

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Estimator
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit.library import ZFeatureMap

# --------------------------------------------------------------------------- #
# QCNN quantum ansatz builder – mirrors the classical QCNN structure
# --------------------------------------------------------------------------- #
def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
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


def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Build a convolutional layer with pairwise two‑qubit blocks."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        block = _conv_circuit(params[i * 3 : (i + 2) * 3])
        qc.compose(block, [i, i + 1], inplace=True)
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unitary."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Build a pooling layer that reduces the qubit count by half."""
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits, 2):
        block = _pool_circuit(params[(i // 2) * 3 : (i // 2 + 1) * 3])
        qc.compose(block, [i, i + 1], inplace=True)
    return qc


def QCNN() -> EstimatorQNN:
    """
    Construct a QCNN ansatz wrapped as an EstimatorQNN.
    The circuit consists of three convolution‑pooling stages followed by a
    feature‑map encoding.
    """
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map
    feature_map = ZFeatureMap(8)

    # Ansatz construction
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(_conv_layer(8, "c1"), inplace=True)
    ansatz.compose(_pool_layer(8, "p1"), inplace=True)
    ansatz.compose(_conv_layer(4, "c2"), inplace=True)
    ansatz.compose(_pool_layer(4, "p2"), inplace=True)
    ansatz.compose(_conv_layer(2, "c3"), inplace=True)
    ansatz.compose(_pool_layer(2, "p3"), inplace=True)

    # Full circuit
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observable – single‑qubit Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    return EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )


# --------------------------------------------------------------------------- #
# FastBaseEstimator – deterministic and shot‑noisy evaluation
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate expectation values of a parametrised quantum circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        circuit:
            The quantum circuit to evaluate.
        shots:
            If provided, use AerSimulator with the given shot count to generate
            noisy expectation values.  ``None`` yields exact Statevector evaluation.
        seed:
            Random seed for the simulator backend.
        """
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._shots = shots
        self._seed = seed
        self._simulator = None
        if shots is not None:
            self._simulator = AerSimulator(seed_simulator=seed, seed_transpiler=seed)

    # ----------------------------------------------------------------------- #
    # Parameter binding
    # ----------------------------------------------------------------------- #
    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    # ----------------------------------------------------------------------- #
    # Core evaluation routine
    # ----------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set.

        Parameters
        ----------
        observables:
            Iterable of Pauli‑operator based observables.
        parameter_sets:
            Sequence of parameter vectors.

        Returns
        -------
        List of rows, each containing the expectation values for the observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if self._shots is None:
            # Exact evaluation via Statevector
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
            return results

        # Shot‑noisy evaluation using AerSimulator
        for values in parameter_sets:
            bound = self._bind(values)
            bound.save_statevector()
            bound.save_expectation_value(
                observable=observables[0], shots=self._shots
            )
            result = self._simulator.run(bound).result()
            meas = result.get_counts()
            # Convert counts to expectation value (simple average of +1/-1)
            exp_val = sum(
                (-1) ** (len(bin(bitstring)[2:].zfill(len(meas))) - 1)
                * count
                for bitstring, count in meas.items()
            ) / self._shots
            results.append([exp_val])
        return results

    # ----------------------------------------------------------------------- #
    # Convenience utilities
    # ----------------------------------------------------------------------- #
    @staticmethod
    def qcnn() -> EstimatorQNN:
        """Return a QCNN EstimatorQNN instance."""
        return QCNN()


__all__ = ["FastBaseEstimator"]
