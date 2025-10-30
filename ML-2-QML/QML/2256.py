import numpy as np
from typing import Sequence, List
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator

def _build_weight_params(arch: Sequence[int]) -> List[Parameter]:
    """Create a list of Parameter objects for all weight values in the architecture."""
    params = []
    for layer_idx, (in_f, out_f) in enumerate(zip(arch[:-1], arch[1:])):
        for out_idx in range(out_f):
            for in_idx in range(in_f):
                params.append(Parameter(f"w_{layer_idx}_{out_idx}_{in_idx}"))
    return params

def _build_input_params(num_qubits: int) -> List[Parameter]:
    """Create input parameters for each qubit."""
    return [Parameter(f"in_{i}") for i in range(num_qubits)]

def _build_circuit(arch: Sequence[int], weight_params: List[Parameter], input_params: List[Parameter]) -> QuantumCircuit:
    """Construct a simple variational circuit that mirrors the classical feed‑forward."""
    num_qubits = arch[0]
    qc = QuantumCircuit(num_qubits)
    # Encode inputs
    for i, p in enumerate(input_params):
        qc.ry(p, i)
    # Apply weight rotations
    idx = 0
    for layer_idx, (in_f, out_f) in enumerate(zip(arch[:-1], arch[1:])):
        for out_idx in range(out_f):
            for in_idx in range(in_f):
                qc.ry(weight_params[idx], out_idx)
                idx += 1
    return qc

def _build_observable(num_outputs: int):
    """Return a list with a single Pauli Y observable on the first output qubit."""
    return [("Y" * num_outputs, 1)]

def build_estimator_qnn(arch: Sequence[int], weight_matrices: List[np.ndarray]) -> EstimatorQNN:
    """Build a Qiskit EstimatorQNN that shares the weight matrices with the classical model."""
    # Flatten weight matrices into a 1‑D array
    flat_weights = np.concatenate([w.flatten() for w in weight_matrices])
    weight_params = _build_weight_params(arch)
    input_params = _build_input_params(arch[0])
    qc = _build_circuit(arch, weight_params, input_params)
    observable = _build_observable(arch[-1])
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    # Set initial weight values
    init_params = {param: val for param, val in zip(weight_params, flat_weights)}
    qnn.set_weight_parameters(init_params)
    return qnn

class UnifiedEstimatorQNN:
    """Quantum estimator that mirrors the classical architecture.

    The class builds a Qiskit EstimatorQNN with the same weight matrices
    used by the classical counterpart.  It exposes a predict method that
    accepts a NumPy array of inputs and returns the expectation value
    of the chosen observable.
    """

    def __init__(self, arch: Sequence[int], weight_matrices: List[np.ndarray]):
        self.arch = arch
        self.qnn = build_estimator_qnn(arch, weight_matrices)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the quantum expectation values for the given inputs."""
        param_dicts = []
        for sample in X:
            param_dict = {f"in_{i}": sample[i] for i in range(len(sample))}
            param_dicts.append(param_dict)
        results = self.qnn.evaluate(param_dicts)
        # Each result is a list of expectation values; we take the first
        return np.array([res[0] for res in results])
