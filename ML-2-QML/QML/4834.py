"""Quantum backbone for EstimatorQNNGen291.

Implements a lightweight 4‑qubit variational ansatz that mirrors the
convolution‑pooling pattern of the QCNN example while staying compatible
with the Quantum‑NAT style random layers.  The circuit accepts a 4‑dimensional
input vector, applies a parameterized rotation per qubit, entangles the
qubits, and returns the expectation values of Y on each qubit."""
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
import numpy as np

class EstimatorQNNGen291Q:
    """
    Quantum variational circuit returning 4 expectation values.
    The circuit is parameterized by 4 input angles and 12 trainable weights.
    """
    def __init__(self):
        # Parameter vectors
        self.input_params = ParameterVector("x", 4)
        self.weight_params = ParameterVector("w", 12)  # 3 per qubit

        # Build circuit
        self.circuit = QuantumCircuit(4)
        # Encode inputs
        for i in range(4):
            self.circuit.ry(self.input_params[i], i)
        # Variational ansatz
        for i in range(4):
            self.circuit.ry(self.weight_params[3*i], i)
            self.circuit.rz(self.weight_params[3*i + 1], i)
            self.circuit.ry(self.weight_params[3*i + 2], i)
        # Entanglement
        self.circuit.cx(0, 1)
        self.circuit.cx(1, 2)
        self.circuit.cx(2, 3)

        # Observables: Y on each qubit
        self.observables = []
        for i in range(4):
            pauli = ["I"] * 4
            pauli[i] = "Y"
            self.observables.append(SparsePauliOp.from_list([("".join(pauli), 1)]))

        # Initialize trainable weights randomly
        self.weights = np.random.uniform(0, 2*np.pi, size=12)

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, 4) containing input angles.

        Returns
        -------
        np.ndarray
            Shape (batch, 4) with expectation values for each qubit.
        """
        batch = inputs.shape[0]
        results = np.zeros((batch, 4))
        # Prepare weight parameter dictionary
        weight_dict = {p: w for p, w in zip(self.weight_params, self.weights)}
        for idx, inpt in enumerate(inputs):
            # Map input angles
            param_dict = {self.input_params[i]: inpt[i] for i in range(4)}
            # Merge with weight parameters
            param_dict.update(weight_dict)
            # Assign parameters to a fresh circuit copy
            circ = self.circuit.copy()
            circ.assign_parameters(param_dict, inplace=True)
            # Compute statevector
            sv = Statevector.from_instruction(circ)
            # Evaluate each observable
            for j, obs in enumerate(self.observables):
                results[idx, j] = sv.expectation_value(obs).real
        return results

__all__ = ["EstimatorQNNGen291Q"]
