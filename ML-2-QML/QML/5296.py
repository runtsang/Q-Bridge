"""Quantum‑only version of the hybrid classifier using a parameter‑driven circuit."""
import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumClassifierModel(nn.Module):
    """
    Quantum circuit‑based classifier that maps a batch of real‑valued feature
    vectors into class logits using a variational ansatz and Pauli‑Z
    expectation values.

    The circuit is built once during initialization and then
    re‑used for every forward pass.  For each input sample the
    parameters of the encoding rotations are bound to the data,
    the circuit is executed on a statevector simulator, and the
    expectation values of a set of Z‑type observables are returned.
    These expectation values are fed into a small classical head to
    produce the final logits.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth

        # Parameter vectors for data encoding and variational layer.
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)

        # Build the circuit once.
        self.circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(self.encoding, range(num_qubits)):
            self.circuit.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                self.circuit.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)

        # Observables for expectation value extraction.
        self.observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        # Default backend; can be overridden by the user.
        self.backend = Aer.get_backend("statevector_simulator")

        # Classical head mapping expectation values to logits.
        self.head = nn.Linear(num_qubits, 2)

    def set_backend(self, backend) -> None:
        """Allow the user to supply a custom backend (e.g., a real quantum device)."""
        self.backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input feature matrix of shape ``(batch, num_qubits)``.
        Returns
        -------
        logits : torch.Tensor
            Logits of shape ``(batch, 2)``.
        """
        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            # Bind the encoding parameters to the current data sample.
            param_dict = {param: float(x[i, j].item()) for j, param in enumerate(self.encoding)}
            bound_circuit = self.circuit.bind_parameters(param_dict)

            # Execute the circuit on the chosen backend.
            job = execute(bound_circuit, self.backend, shots=1024)
            result = job.result()
            statevector = result.get_statevector(bound_circuit)

            # Compute expectation values for each observable.
            exp_vals = []
            for op in self.observables:
                mat = op.to_matrix()
                exp = np.real(np.vdot(statevector, mat @ statevector))
                exp_vals.append(exp)

            outputs.append(exp_vals)

        # Convert to a torch tensor and feed through the classical head.
        exp_tensor = torch.tensor(outputs, dtype=torch.float32, device=x.device)
        logits = self.head(exp_tensor)
        return logits


__all__ = ["QuantumClassifierModel"]
