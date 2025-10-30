"""Hybrid quantum‑classical estimator that mirrors the classical
EstimatorQNNHybrid.  The quantum circuit implements the following
stages:

1. **Input encoding** – each input feature is encoded with a Ry
   rotation on a dedicated qubit.
2. **Random layer** – a deterministic sequence of Ry and CX gates that
   emulates a quantum kernel.
3. **Self‑attention block** – a 4‑qubit rotation‑entangle sub‑circuit
   that mimics the quantum self‑attention described in the reference.
4. **Read‑out** – all qubits are measured in the Z basis and the
   expectation values are linearly combined to produce the final
   regression output.

The implementation uses the Aer QASM simulator and the
StatevectorEstimator primitive.  Because the circuit is fully
parameterised it can be trained with a gradient‑based optimiser
(e.g. Qiskit’s AerOptimizer).
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class EstimatorQNNHybrid:
    """
    Quantum implementation of the hybrid estimator.
    The class exposes a ``run`` method that accepts a batch of input
    values and returns the expectation value of the observable defined
    on the final qubits.
    """

    def __init__(self,
                 input_dim: int = 2,
                 n_qubits: int = 8) -> None:
        self.input_dim = input_dim
        self.n_qubits = n_qubits

        # Parameters
        self.input_params = [Parameter(f"x{i}") for i in range(input_dim)]
        self.weight_params = [Parameter(f"w{i}") for i in range(n_qubits)]

        # Build the base circuit
        self.circuit = QuantumCircuit(n_qubits)

        # 1. Encode inputs on the first ``input_dim`` qubits
        for i, p in enumerate(self.input_params):
            self.circuit.ry(p, i)

        # 2. Random layer (quantum kernel) on all qubits
        self._apply_random_layer(self.circuit, self.weight_params)

        # 3. Self‑attention block
        self._build_attention(self.circuit)

        # 4. Measurement
        self.circuit.measure_all()

        # Observable for expectation value (Z on each qubit)
        self.observables = [Pauli('Z') for _ in range(n_qubits)]

        # Estimator primitive
        backend = Aer.get_backend("statevector_simulator")
        estimator = StatevectorEstimator(backend=backend)

        # Wrap into EstimatorQNN for convenient batch evaluation
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=estimator,
        )

    def _apply_random_layer(self, circuit: QuantumCircuit,
                            params: list[Parameter]) -> None:
        """Append a deterministic random layer to the circuit."""
        for i, p in enumerate(params):
            circuit.ry(p, i)
            if i < len(params) - 1:
                circuit.cx(i, i + 1)

    def _build_attention(self, circuit: QuantumCircuit) -> None:
        """Append a simple rotation‑entangle block."""
        for i in range(self.n_qubits):
            circuit.rx(np.pi / 4, i)
            circuit.ry(np.pi / 4, i)
            circuit.rz(np.pi / 4, i)
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, input_dim) with values in [0, 2π].

        Returns
        -------
        np.ndarray
            Shape (batch,) containing the expectation value of the
            observable as a scalar.
        """
        # The EstimatorQNN run method expects a 2‑D array of shape
        # (batch, len(input_params)) and returns a 1‑D array of
        # expectation values.
        expectations = self.estimator_qnn.run(inputs)
        # Convert to a plain numpy array of real values
        return np.asarray(expectations).real

__all__ = ["EstimatorQNNHybrid"]
