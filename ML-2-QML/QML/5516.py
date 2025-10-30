import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class QCNNHybrid(nn.Module):
    """Quantum‑equivalent of the classical QCNNHybrid. The convolutional
    and fully‑connected blocks are implemented as a quantum circuit
    using a feature map and a layered ansatz. The final measurement
    returns a scalar expectation that is interpreted as a probability
    after a sigmoid.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz (default 4).
    shots : int
        Number of shots for measurement (unused with StatevectorEstimator).
    """
    def __init__(self, num_qubits: int = 4, shots: int = 100):
        super().__init__()
        self.num_qubits = num_qubits
        self.shots = shots

        # Build the quantum circuit
        self.circuit = self._build_circuit()

        # Estimator for expectation values
        self.estimator = StatevectorEstimator()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    def _build_circuit(self) -> QuantumCircuit:
        # Feature map: encode 4‑dim input into rotation angles
        feature_params = ParameterVector("x", length=self.num_qubits)
        qc = QuantumCircuit(self.num_qubits)
        qc.h(range(self.num_qubits))
        qc.ry(feature_params, range(self.num_qubits))

        # Ansatz: simple layered circuit mimicking conv + pool operations
        ansatz_params = ParameterVector("a", length=self.num_qubits * 2)
        for i in range(0, self.num_qubits, 2):
            qc.cx(i, i + 1)
            qc.rz(ansatz_params[i], i)
            qc.ry(ansatz_params[i + 1], i + 1)
            qc.cx(i + 1, i)

        qc.measure_all()
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accept a batch of 4‑dim feature vectors and return a probability
        vector of shape (batch, 1)."""
        batch = x.shape[0]
        expectations = []

        for i in range(batch):
            # Bind feature parameters to the input vector
            binding = {self.circuit.parameters[j]: x[i, j].item()
                       for j in range(self.num_qubits)}
            # Fix ansatz parameters to zero (trainable in a full training loop)
            for j in range(self.num_qubits, len(self.circuit.parameters)):
                binding[self.circuit.parameters[j]] = 0.0

            result = self.estimator.run(
                circuits=[self.circuit],
                parameter_binds=[binding],
                observables=[self.observable]
            )
            expectations.append(result.values[0])

        expectations = torch.tensor(expectations, dtype=torch.float32)
        # Convert to probability with sigmoid
        probs = torch.sigmoid(expectations)
        return probs

    def hybrid(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper to match the API of the classical module."""
        return self.forward(x)

__all__ = ["QCNNHybrid"]
