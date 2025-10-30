"""
ConvHybrid – quantum implementation with a variational filter and classifier circuit.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a simple layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input dimensions.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The full parameterized circuit.
    encoding : Iterable
        Parameter vector for data encoding.
    weights : Iterable
        Parameter vector for variational weights.
    observables : List[SparsePauliOp]
        Pauli‑Z observables for each qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Measurements
    circuit.measure_all()

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


class ConvHybrid:
    """
    Quantum variational filter that accepts 2×2 image patches and returns a
    probability‑based feature.  It mirrors the classical interface: :meth:`run`
    performs a circuit execution and :meth:`classify` runs the classifier ansatz.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the image patch (must be square).
    threshold : float, default 0.5
        Threshold used to encode classical values into qubit angles.
    shots : int, default 100
        Number of shots for the simulator.
    backend : qiskit.providers.BaseBackend | None, default None
        Backend to execute the circuit; defaults to Aer qasm_simulator.
    depth : int, default 2
        Depth of the variational classifier ansatz.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.5, shots: int = 100,
                 backend: qiskit.providers.BaseBackend | None = None, depth: int = 2) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.n_qubits = kernel_size ** 2

        # Build filter circuit (same as classical conv but quantum)
        self.filter_circuit = self._build_filter_circuit()

        # Build classifier ansatz
        self.classifier_circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits=self.n_qubits, depth=depth
        )

    def _build_filter_circuit(self) -> QuantumCircuit:
        """Internal helper that produces a simple RX‑based encoding circuit."""
        circuit = QuantumCircuit(self.n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            circuit.rx(theta[i], i)
        circuit.barrier()
        circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        circuit.measure_all()
        return circuit

    def run(self, data: np.ndarray) -> float:
        """
        Execute the filter circuit on a single 2×2 patch.

        Parameters
        ----------
        data : np.ndarray
            2×2 array with values in [0, 1] (or arbitrary, thresholded).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.filter_circuit.parameters[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self.filter_circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.filter_circuit)

        # Compute weighted sum of |1> counts
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

    def classify(self, data: np.ndarray) -> np.ndarray:
        """
        Run the classifier ansatz on a 2×2 patch and return raw expectation values.

        Parameters
        ----------
        data : np.ndarray
            2×2 array with values in [0, 1] (or arbitrary, thresholded).

        Returns
        -------
        np.ndarray
            Expectation values for each qubit observable.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.encoding[i]] = val
                bind[self.weights[i]] = 0  # initialize weights to zero for evaluation
            param_binds.append(bind)

        job = execute(
            self.classifier_circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.classifier_circuit)

        # Convert counts to expectation values of Z
        expectations = np.zeros(self.n_qubits)
        for key, val in result.items():
            for i, bit in enumerate(reversed(key)):
                expectations[i] += ((-1) ** int(bit)) * val

        expectations /= self.shots
        return expectations

    def get_weight_sizes(self) -> List[int]:
        """Return the number of parameters per variational layer."""
        return [self.weights[i].size for i in range(len(self.weights))]


__all__ = ["ConvHybrid", "build_classifier_circuit"]
