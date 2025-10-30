"""Hybrid classifier that uses a classical autoencoder to produce latent
vectors and a variational quantum circuit to compute class logits.

The class :class:`HybridClassifierAutoencoder` mirrors the classical
implementation but replaces the linear head with a Qiskit circuit that
accepts the latent vector as parameters.  The circuit is built using the
ansatz from the reference pair: a layer of RX for encoding, followed by
RY rotations and CZ entangling gates.  The output is the expectation
value of Z on each qubit, which is interpreted as a logit.

The module is fully compatible with the original QuantumClassifierModel
factory while providing a quantum‑only implementation of the classifier.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple, List

from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from.Autoencoder import Autoencoder

__all__ = ["HybridClassifierAutoencoder"]

def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector, List[SparsePauliOp]]:
    """Construct a variational circuit that mirrors the quantum helper.

    Parameters
    ----------
    num_qubits : int
        Number of qubits, equal to the latent dimensionality.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The variational ansatz.
    encoding : ParameterVector
        Parameters that encode the classical latent vector.
    weights : ParameterVector
        Variational parameters of the ansatz.
    observables : List[SparsePauliOp]
        Pauli‑Z observables on each qubit used as logits.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, encoding, weights, observables

class HybridClassifierAutoencoder:
    """Quantum‑backed hybrid classifier.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw input vector.
    latent_dim : int, default 32
        Size of the latent space produced by the autoencoder.
    hidden_dims : Tuple[int,...], default (128, 64)
        Hidden layer sizes for the autoencoder encoder/decoder.
    dropout : float, default 0.1
        Drop‑out probability in the autoencoder.
    depth : int, default 3
        Depth of the variational circuit.
    shots : int, default 1024
        Number of shots used to estimate expectation values.
    backend : Backend, optional
        Qiskit backend; defaults to Aer qasm simulator.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int,...] = (128, 64),
        dropout: float = 0.1,
        depth: int = 3,
        shots: int = 1024,
        backend=None,
    ) -> None:
        self.encoder = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.depth = depth
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        (
            self.circuit,
            self.encoding_params,
            self.weight_params,
            self.observables,
        ) = build_classifier_circuit(latent_dim, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the input through the encoder and the variational circuit."""
        latent = self.encoder.encode(x).detach().cpu().numpy()
        param_binds = [{self.encoding_params[i]: latent[i] for i in range(len(latent))}]
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        expectations = []
        for qubit in range(len(self.observables)):
            exp_val = 0.0
            for bitstring, freq in counts.items():
                # Qiskit encodes qubit 0 as the right‑most bit.
                bit = int(bitstring[::-1][qubit])
                exp_val += ((-1) ** bit) * freq
            exp_val /= self.shots
            expectations.append(exp_val)

        return torch.tensor(expectations, dtype=torch.float32)
