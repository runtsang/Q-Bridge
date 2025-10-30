import numpy as np
from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RX, RY, RZ, CZ
from qiskit.opflow import PauliSumOp, StateFn, CircuitStateFn, Gradient, AerPauliExpectation, PauliExpectation
from qiskit.providers.aer import AerSimulator
from qiskit.opflow import PauliTrotterEvolution
from qiskit.opflow import StateFn, ExpectationFactory
from qiskit.opflow import OperatorBase
from qiskit.opflow import PauliExpectation
from qiskit.opflow import Gradient

class QuantumClassifierModel:
    """
    Quantum classifier factory with configurable encoding, ansatz depth,
    and measurement operators.  Implements a simple variational circuit
    and exposes methods for parameter extraction, expectation evaluation,
    and gradient computation using the parameter‑shift rule.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features) in the classifier.
    depth : int
        Number of ansatz layers.
    encoding : str, optional
        Encoding type: 'rx' (default) or 'ry'.  Allows comparison with
        classical angular embeddings.
    dropout : float, optional
        Probability of dropping qubits during training (used only for
        data‑augmentation in simulation).
    """
    def __init__(
        self,
        num_qubits: int,
        depth: int,
        encoding: str = "rx",
        dropout: float = 0.0,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = encoding
        self.dropout = dropout

        # build circuit and store parameters
        self.circuit, self.encoding_params, self.weights, self.observables = self._build_circuit()

        # simulator for expectation evaluation
        self.backend = AerSimulator(method="statevector")

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """
        Create a layered ansatz with optional data‑encoding gates.
        Returns the circuit, encoding param vector, variational weight vector,
        and measurement operators (one Z per qubit).
        """
        # Parameters for data encoding
        x = ParameterVector("x", self.num_qubits)
        # Parameters for variational weights
        theta = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        # Encoding
        if self.encoding == "rx":
            for q, param in enumerate(x):
                qc.rx(param, q)
        else:
            for q, param in enumerate(x):
                qc.ry(param, q)

        # Ansatz layers
        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qc.ry(theta[idx], q)
                idx += 1
            for q in range(self.num_qubits - 1):
                qc.cz(q, q + 1)

        # Observables: Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return qc, [x], [theta], observables

    def get_metadata(self) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """
        Return the circuit, encoding/weight parameters, and observables.
        """
        return self.circuit, self.encoding_params, self.weights, self.observables

    def evaluate_expectation(self, data: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Compute expectation values for the observables given input data and
        variational parameters.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, num_qubits)
            Input feature matrix (real numbers).
        params : np.ndarray, shape (num_params,)
            Variational parameters (theta).
        Returns
        -------
        np.ndarray, shape (n_samples, num_qubits)
            Expectation values of each observable.
        """
        # Bind parameters
        bound_circuit = self.circuit.bind_parameters(
            {p: v for p, v in zip(self.encoding_params[0], data.T)}
        )
        bound_circuit = bound_circuit.bind_parameters(
            {p: v for p, v in zip(self.weights[0], params)}
        )
        # Run simulation
        job = self.backend.run(bound_circuit, shots=1024)
        result = job.result()
        exp_vals = []
        for obs in self.observables:
            exp = result.get_expectation_value(obs)
            exp_vals.append(exp)
        return np.array(exp_vals).T

    def parameter_shift_grad(self, data: np.ndarray, params: np.ndarray, shift: float = np.pi / 2) -> np.ndarray:
        """
        Estimate gradient of the expectation vector w.r.t. variational parameters
        using the parameter‑shift rule.
        """
        grads = np.zeros_like(params)
        for i in range(len(params)):
            shift_vec = np.zeros_like(params)
            shift_vec[i] = shift
            exp_plus = self.evaluate_expectation(data, params + shift_vec)
            exp_minus = self.evaluate_expectation(data, params - shift_vec)
            grads[i] = np.mean(exp_plus - exp_minus, axis=0)
        return grads

    def train_one_step(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        params: np.ndarray,
        lr: float = 0.01,
    ) -> Tuple[np.ndarray, float]:
        """
        Perform a single gradient‑descent step on the variational parameters.
        The cost function is mean‑squared‑error between expectation values
        and one‑hot encoded labels.

        Returns
        -------
        new_params : np.ndarray
            Updated parameters.
        loss : float
            Loss value after the step.
        """
        exp = self.evaluate_expectation(data, params)
        loss = np.mean((exp - labels) ** 2)
        grads = self.parameter_shift_grad(data, params)
        new_params = params - lr * grads.mean(axis=0)
        return new_params, loss

__all__ = ["QuantumClassifierModel"]
