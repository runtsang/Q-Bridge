from __future__ import annotations

import torch
import torch.nn as nn
from qiskit.circuit import Parameter
from qiskit import Aer, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator

class EstimatorQNNModel(nn.Module):
    """Quantum variational circuit that maps 4 input features to 4 expectation values."""
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # Input parameters (one per qubit)
        self.input_params = [Parameter(f"input_{i}") for i in range(n_qubits)]
        # Weight parameters (3 per qubit: RX, RY, RZ)
        self.weight_params = [Parameter(f"weight_{i}") for i in range(n_qubits * 3)]

        # Build the circuit
        qc = QuantumCircuit(n_qubits)
        # Encode inputs with RY rotations
        for i in range(n_qubits):
            qc.ry(self.input_params[i], i)
        # Variational layer: parameterized rotations
        for i in range(n_qubits):
            qc.rx(self.weight_params[3 * i], i)
            qc.ry(self.weight_params[3 * i + 1], i)
            qc.rz(self.weight_params[3 * i + 2], i)
        # Entangling layer: a simple CNOT chain
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

        # Observables: Pauli‑Z on each qubit
        observables = []
        for i in range(n_qubits):
            pauli_str = "I" * i + "Z" + "I" * (n_qubits - i - 1)
            observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))

        # Backend estimator
        backend = Aer.get_backend("statevector_simulator")
        estimator = QiskitEstimator(backend=backend)

        # Wrap into a Qiskit EstimatorQNN (torch.nn.Module)
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=estimator,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Build parameter dictionary for the batch
        param_dict = {self.input_params[i]: inputs[:, i] for i in range(self.n_qubits)}
        # Evaluate the quantum circuit
        expectations = self.estimator_qnn(param_dict)  # numpy array of shape (batch, n_qubits)
        return torch.tensor(expectations, dtype=torch.float32, device=inputs.device)

__all__ = ["EstimatorQNNModel"]
