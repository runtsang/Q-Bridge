import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class HybridNATModel:
    """
    Quantum hybrid model that encodes classical features, applies a
    self‑attention‑style entangling circuit, and evaluates a variational
    EstimatorQNN.  The class is fully compatible with the classical
    counterpart and can be used in end‑to‑end quantum‑classical experiments.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        # Input (feature) parameters
        self.input_params = [Parameter(f"input{i}") for i in range(n_qubits)]
        # Weight parameters for variational gates
        self.weight_params = [Parameter(f"weight{i}") for i in range(n_qubits)]
        # Entanglement parameters for self‑attention
        self.entangle_params = [Parameter(f"entangle{i}") for i in range(n_qubits - 1)]

        # Build the variational circuit
        self.circuit = QuantumCircuit(n_qubits)
        # Feature encoding (Ry for each qubit)
        for i in range(n_qubits):
            self.circuit.h(i)
            self.circuit.ry(self.input_params[i], i)
            self.circuit.rx(self.weight_params[i], i)
        # Self‑attention style entanglement (CRX)
        for i in range(n_qubits - 1):
            self.circuit.crx(self.entangle_params[i], i, i + 1)

        # Observable: Pauli‑Y on all qubits
        self.observable = SparsePauliOp.from_list([("Y" * n_qubits, 1)])

        # Estimator primitive
        self.estimator = StatevectorEstimator()
        # EstimatorQNN wrapper
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params + self.entangle_params,
            estimator=self.estimator,
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the EstimatorQNN for a batch of classical inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (batch, n_qubits) containing values for the
            input parameters.  Each column corresponds to a qubit.

        Returns
        -------
        np.ndarray
            Array of shape (batch,) with the predicted scalar outputs.
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        batch = inputs.shape[0]
        results = []
        for i in range(batch):
            param_dict = {p: inputs[i, idx] for idx, p in enumerate(self.input_params)}
            pred = self.estimator_qnn.predict(param_dict)[0]
            results.append(pred)
        return np.array(results)

    def parameters(self):
        """
        Return all trainable parameters for external optimization.

        Returns
        -------
        list
            List of Parameter objects.
        """
        return self.weight_params + self.entangle_params


__all__ = ["HybridNATModel"]
