"""Quantum QCNN with noise simulation and parameter‑shift gradient support."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import SPSA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.utils import QuantumInstance
from qiskit import Aer


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unit from the original design."""
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
    """Two‑qubit pooling unit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Convolutional layer composed of pairwise conv circuits."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.append(_conv_circuit(params[i:i+3]), [qubits[i], qubits[i+1]])
        qc.barrier()
    return qc


def _pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Pooling layer that merges source qubits into sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, sink in zip(sources, sinks):
        qc.append(_pool_circuit(params[3 * sources.index(src):3 * (sources.index(src)+1)]),
                  [src, sink])
        qc.barrier()
    return qc


def QCNN(noise: bool = False, noise_level: float = 0.01) -> EstimatorQNN:
    """
    Build a QCNN ansatz with optional depolarising noise and a parameter‑shift gradient estimator.

    Parameters
    ----------
    noise : bool
        If True, a simple depolarising noise model is applied to all two‑qubit gates.
    noise_level : float
        Depolarising probability per gate when ``noise=True``.

    Returns
    -------
    EstimatorQNN
        A Qiskit EstimatorQNN ready for hybrid training.
    """
    algorithm_globals.random_seed = 12345
    backend = Aer.get_backend("aer_simulator_statevector")

    if noise:
        # Simple two‑qubit depolarising error for all CNOTs
        noise_model = NoiseModel()
        error = depolarizing_error(noise_level, 2)
        noise_model.add_all_qubit_quantum_error(error, ["cx"])
        quantum_instance = QuantumInstance(backend=backend, noise_model=noise_model)
    else:
        quantum_instance = QuantumInstance(backend=backend)

    # Feature map
    feature_map = ZFeatureMap(8)
    # Ansatz construction
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First conv + pool
    ansatz.compose(_conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(_pool_layer([0,1,2,3], [4,5,6,7], "p1"), range(8), inplace=True)

    # Second conv + pool
    ansatz.compose(_conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(_pool_layer([0,1], [2,3], "p2"), range(4, 8), inplace=True)

    # Third conv + pool
    ansatz.compose(_conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(_pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    # Observable
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # EstimatorQNN with parameter‑shift gradient
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=StatevectorEstimator(quantum_instance=quantum_instance),
        gradient_method="parameter-shift",
    )
    return qnn


__all__ = ["QCNN"]
