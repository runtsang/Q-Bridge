"""Hybrid quantum layer that reproduces the classical HybridLayer behaviour.

The circuit consists of:
    * a parameterised rotation block (RY+RX) per qubit – analogous to the FC layer
    * controlled‑RX entanglement – mirroring the self‑attention pattern
    * a Y‑Pauli observable on each qubit
The EstimatorQNN wrapper turns the circuit into a differentiable QNN that can be used
with classical optimisers.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class HybridLayer:
    """
    Quantum implementation of the hybrid classical block.
    The run method accepts three numpy arrays:
        * input_vals – values for the input rotation parameters
        * weight_vals – values for the weight rotation parameters
        * entangle_vals – values for the controlled‑RX gates
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        # Define parameters
        self.input_params = [Parameter(f"x{i}") for i in range(n_qubits)]
        self.weight_params = [Parameter(f"w{i}") for i in range(n_qubits)]
        self.entangle_params = [Parameter(f"e{i}") for i in range(n_qubits - 1)]

        # Build circuit
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        for i in range(n_qubits):
            self.circuit.ry(self.input_params[i], i)
            self.circuit.rx(self.weight_params[i], i)
        for i in range(n_qubits - 1):
            self.circuit.crx(self.entangle_params[i], i, i + 1)
        self.circuit.measure_all()

        # Observable: Y on all qubits
        self.observable = SparsePauliOp.from_list([("Y" * n_qubits, 1.0)])

        # EstimatorQNN wrapper
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def run(self, input_vals: np.ndarray, weight_vals: np.ndarray, entangle_vals: np.ndarray) -> np.ndarray:
        """
        Execute the quantum circuit with the given parameters and return the expectation value.
        """
        bind_dict = {}
        for i in range(self.n_qubits):
            bind_dict[self.input_params[i]] = input_vals[i]
            bind_dict[self.weight_params[i]] = weight_vals[i]
        for i in range(self.n_qubits - 1):
            bind_dict[self.entangle_params[i]] = entangle_vals[i]
        # EstimatorQNN expects a list of bindings
        preds = self.estimator_qnn.predict([bind_dict])
        return np.array(preds[0])

__all__ = ["HybridLayer"]
