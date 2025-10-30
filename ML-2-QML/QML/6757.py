from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.providers.aer import AerSimulator


def build_regression_circuit(num_qubits: int, depth: int):
    """Construct a layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input dimension.
    depth : int
        Depth of the variational ansatz.

    Returns
    -------
    circuit : QuantumCircuit
        The variational circuit with symbolic parameters.
    encoding : ParameterVector
        Parameters that encode the classical input.
    weights : ParameterVector
        Trainable variational parameters.
    observables : list[SparsePauliOp]
        Z observables on each qubit for feature extraction.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, encoding, weights, observables


class HybridRegressionQML(nn.Module):
    """Quantum‑classical hybrid regression model that uses a variational circuit
    followed by a linear head.  The circuit is identical to the one used in the
    classifier example but the measurement is the expectation value of Z on each
    qubit, producing a real‑valued feature vector for the classical head.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input dimension.
    depth : int, default 1
        Depth of the variational ansatz.
    backend : qiskit.providers.BaseBackend, optional
        Backend used to simulate the circuit.  If ``None`` an AerSimulator is
        instantiated.
    """
    def __init__(self, num_qubits: int, depth: int = 1, backend=None):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or AerSimulator()
        self.circuit, self.encoding_params, self.weight_params, self.observables = build_regression_circuit(
            num_qubits, depth
        )

        # Trainable variational weights as a flat torch parameter vector
        self.weight_vector = nn.Parameter(torch.randn(len(self.weight_params)))

        # Classical linear head
        self.head = nn.Linear(num_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_qubits) containing rotation angles for
            the encoding RX gates.

        Returns
        -------
        torch.Tensor
            Regression predictions of shape (batch,).
        """
        batch_size = x.shape[0]
        preds = []

        for i in range(batch_size):
            # Build a ParameterResolver that binds both encoding angles and variational weights
            param_resolver = {p: x[i, idx].item() for idx, p in enumerate(self.encoding_params)}
            param_resolver.update(
                {p: self.weight_vector[idx].item() for idx, p in enumerate(self.weight_params)}
            )

            circ = self.circuit.assign_parameters(param_resolver)

            # Run the circuit on the simulator and obtain the statevector
            result = self.backend.run(circ, shots=0).result()
            statevec = result.get_statevector()

            # Compute expectation values of Z on each qubit
            evs = []
            sv = Statevector(statevec)
            for obs in self.observables:
                ev = np.real(sv.expectation_value(obs))
                evs.append(ev)
            evs = torch.tensor(evs, dtype=torch.float32, device=x.device)
            preds.append(evs)

        features = torch.stack(preds, dim=0)
        return self.head(features).squeeze(-1)

    def set_weights(self, weights: torch.Tensor) -> None:
        """Utility to set the trainable variational weights."""
        self.weight_vector.data.copy_(weights)

    def get_weights(self) -> torch.Tensor:
        """Return the current variational weights."""
        return self.weight_vector.detach()


__all__ = ["HybridRegressionQML"]
