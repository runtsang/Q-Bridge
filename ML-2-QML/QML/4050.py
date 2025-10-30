"""Quantum circuit helper for the hybrid classifier.

Provides a class QuantumHybridClassifier that builds a parameterised
variational circuit and exposes a ``run`` method to compute
expectation values on a Qiskit simulator.  The circuit supports
data encoding, a random layer, and multi‑qubit Pauli‑Z readout.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np

def _build_circuit(num_qubits: int,
                   depth: int,
                   use_random_layer: bool = True) -> Tuple[QuantumCircuit,
                                                           Iterable,
                                                           Iterable,
                                                           List[SparsePauliOp]]:
    """
    Internal helper that creates the circuit.
    """
    encoding = ParameterVector("x", num_qubits)
    weight = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Optional random layer (30 ops)
    if use_random_layer:
        np.random.seed(42)
        for _ in range(30):
            target = np.random.randint(num_qubits)
            control = np.random.randint(num_qubits)
            if control!= target:
                circuit.cx(control, target)
            circuit.ry(np.random.uniform(0, 2 * np.pi), target)

    # Variational ansatz
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weight[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return circuit, encoding, weight, observables

class QuantumHybridClassifier:
    """
    Quantum part of the hybrid classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    depth : int
        Depth of the variational ansatz.
    backend_name : str
        Qiskit backend name.
    shots : int
        Number of shots for simulation.
    """
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 backend_name: str = "qasm_simulator",
                 shots: int = 1024):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = Aer.get_backend(backend_name)
        self.shots = shots
        self.circuit, self.encoding_params, self.weight_params, self.observables = _build_circuit(num_qubits, depth)

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the circuit on a batch of data.

        Parameters
        ----------
        data : np.ndarray of shape (batch, num_qubits)
            Rotation angles in radians.

        Returns
        -------
        np.ndarray of shape (batch, num_qubits)
            Expectation values of Pauli‑Z on each qubit.
        """
        batch_size = data.shape[0]
        param_binds = []
        for i in range(batch_size):
            bind = {param: data[i, j] for j, param in enumerate(self.encoding_params)}
            for w in self.weight_params:
                bind[w] = 0.0  # initialise variational params to zero
            param_binds.append(bind)

        job = qiskit.execute(self.circuit,
                             backend=self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result()
        expectations = np.array([result.get_expectation_value(obs, self.circuit)
                                 for obs in self.observables])
        return expectations

__all__ = ["QuantumHybridClassifier"]
