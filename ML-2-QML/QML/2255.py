"""Hybrid estimator for quantum circuits with optional shot noise and QCNN QNN support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Union

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class FastHybridEstimator:
    """
    Evaluate a quantum circuit or a QCNN QNN for batches of parameters and observables.

    Parameters
    ----------
    circuit : QuantumCircuit | EstimatorQNN
        The quantum object to evaluate.  ``EstimatorQNN`` is used for the QCNN
        quantum neural network; otherwise a raw ``QuantumCircuit`` is assumed.
    noise_shots : int | None, default=None
        If provided, adds complex Gaussian noise with variance ``1/shots`` to each
        expectation value to emulate shot noise.
    noise_seed : int | None, default=None
        Seed for the pseudo‑random number generator used for noise.
    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, EstimatorQNN],
        noise_shots: int | None = None,
        noise_seed: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.noise_shots = noise_shots
        self.noise_seed = noise_seed
        if not isinstance(circuit, (QuantumCircuit, EstimatorQNN)):
            raise TypeError("circuit must be a QuantumCircuit or EstimatorQNN")

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute the expectation value of each observable for every parameter set.

        For a plain ``QuantumCircuit`` the circuit is bound with the supplied
        parameters and the statevector is used to compute the expectation
        values.  For an ``EstimatorQNN`` the built‑in predict routine is
        leveraged.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if isinstance(self.circuit, EstimatorQNN):
            # EstimatorQNN handles its own parameters internally; we simply
            # forward the input parameters as a batch.
            for params in parameter_sets:
                # ``predict`` expects a NumPy array of shape (n_samples, n_features)
                preds = self.circuit.predict(inputs=np.array([params]), batch_size=None)
                results.append(preds.tolist()[0])
        else:
            for params in parameter_sets:
                state = Statevector.from_instruction(self._bind(params))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)

        if self.noise_shots is not None:
            rng = np.random.default_rng(self.noise_seed)
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [
                    complex(
                        rng.normal(np.real(val), max(1e-6, 1 / self.noise_shots))
                        + 1j
                        * rng.normal(np.imag(val), max(1e-6, 1 / self.noise_shots))
                    )
                    for val in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Bind a list of parameters to the underlying circuit."""
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)


# --------------------------------------------------------------------------- #
# QCNN quantum neural network construction
# --------------------------------------------------------------------------- #
def QCNN() -> EstimatorQNN:
    """
    Build the QCNN quantum neural network using a feature map, convolutional
    and pooling layers, and return an ``EstimatorQNN`` instance.
    """
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

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(pool_circuit(params[param_index : param_index + 3]), [source, sink])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc

    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)

    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)

    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)

    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Combine feature map and ansatz
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
