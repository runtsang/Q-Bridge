"""Core circuit factory for the incremental data-uploading classifier.

Enhancements:
- Data reuploading feature map with parameterized RX, RZ rotations.
- Layered ansatz with optional block-wise entanglement.
- Batch-enabled forward pass using a Qiskit Aer simulator.
- Fidelity-based loss accessor for hybrid training loops.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np

class QClassifier:
    """Hybrid quantum classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational layers.
    backend : str, optional
        Name of the Qiskit simulator backend (default: 'qasm_simulator').
    shots : int, optional
        Number of shots per evaluation (default: 1024).
    """
    def __init__(
        self,
        num_qubits: int,
        depth: int,
        backend: str = "qasm_simulator",
        shots: int = 1024,
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend_name = backend
        self.shots = shots
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.backend = Aer.get_backend(self.backend_name)
        # Initialize weights to a small random seed
        self.weight_values = np.random.uniform(-np.pi, np.pi, size=len(self.weights))

    def set_weights(self, values: np.ndarray) -> None:
        """Set the variational parameters used during forward evaluation."""
        if values.shape!= (len(self.weights),):
            raise ValueError("Weight vector has incorrect shape.")
        self.weight_values = values

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the circuit on a batch of inputs.

        Parameters
        ----------
        inputs
            Array of shape ``(batch, num_qubits)`` with feature values.

        Returns
        -------
        probs
            Array of shape ``(batch, 2)`` with class probabilities.
        """
        probs = np.zeros((inputs.shape[0], 2))
        for i, x in enumerate(inputs):
            param_dict = {p: val for p, val in zip(self.encoding, x)}
            for w, val in zip(self.weights, self.weight_values):
                param_dict[w] = val
            bound_circ = self.circuit.bind_parameters(param_dict)
            job = execute(bound_circ, backend=self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            # Convert counts to probability of measuring each observable
            # For simplicity, we use the first two Z observables.
            prob_0 = 0.0
            prob_1 = 0.0
            for bitstring, cnt in counts.items():
                if bitstring[-1] == "0":
                    prob_0 += cnt
                else:
                    prob_1 += cnt
            probs[i, 0] = prob_0 / self.shots
            probs[i, 1] = prob_1 / self.shots
        return probs

    def fidelity_loss(self, target: np.ndarray, preds: np.ndarray) -> float:
        """Compute a simple fidelity-based loss.

        Parameters
        ----------
        target
            One-hot encoded target labels of shape ``(batch, 2)``.
        preds
            Predicted class probabilities of shape ``(batch, 2)``.

        Returns
        -------
        loss
            Mean fidelity loss over the batch.
        """
        # Fidelity between two probability distributions
        fidelity = np.sum(np.sqrt(target * preds), axis=1)
        return 1.0 - np.mean(fidelity)

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a data‑reuploading variational circuit with entanglement.

    Parameters
    ----------
    num_qubits
        Number of qubits / input features.
    depth
        Number of variational layers.

    Returns
    -------
    circuit
        A ``QuantumCircuit`` instance ready for parameter binding.
    encoding
        Iterable of ParameterVector objects for the data encoding.
    weights
        Iterable of ParameterVector objects for the variational parameters.
    observables
        List of Pauli observables used for the output readout.
    """
    encoding = ParameterVector("x", length=num_qubits)
    weights = ParameterVector("theta", length=num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data re‑uploading: encode data before each variational layer
    for layer in range(depth):
        # Feature map: RX + RZ rotations
        for q in range(num_qubits):
            circuit.rx(encoding[q], q)
            circuit.rz(encoding[q], q)

        # Variational layer
        idx = layer * num_qubits
        for q in range(num_qubits):
            circuit.ry(weights[idx + q], q)

        # Entanglement: a ring of CZ gates
        for q in range(num_qubits):
            circuit.cz(q, (q + 1) % num_qubits)

    # Readout: measure each qubit in the Z basis
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]

    return circuit, list(encoding), list(weights), observables

__all__ = ["build_classifier_circuit", "QClassifier"]
