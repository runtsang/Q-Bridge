"""Quantum‑centric implementation of the HybridTransformerClassifier.

The class builds a parameter‑encoded quantum circuit that mimics the
transformer structure: a token‑encoding layer, a stack of
variational blocks (attention‑like and feed‑forward‑like) and a
measurement head.  The forward method executes the circuit on a
Qiskit simulator and returns the probability of the |1> outcome for
each class qubit.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# --------------------------------------------------------------------------- #
# 1. Quantum building blocks – encoding, attention‑like, feed‑forward‑like
# --------------------------------------------------------------------------- #

def _token_encoding(num_qubits: int) -> ParameterVector:
    """RX encoding of classical token values."""
    return ParameterVector("x", num_qubits)

def _attention_layer(num_qubits: int, depth: int) -> QuantumCircuit:
    """Variational layer that implements an attention‑like mix of rotations
    and entangling gates."""
    qc = QuantumCircuit(num_qubits)
    theta = ParameterVector("theta", num_qubits * depth)
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(theta[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
    return qc

def _ffn_layer(num_qubits: int, depth: int) -> QuantumCircuit:
    """Two‑layer feed‑forward style circuit using rotations and CNOTs."""
    qc = QuantumCircuit(num_qubits)
    phi = ParameterVector("phi", num_qubits * depth)
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.rx(phi[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
    return qc

# --------------------------------------------------------------------------- #
# 2. Classifier circuit factory – builds a complete circuit
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_qubits: int,
                             num_blocks: int,
                             ffn_depth: int,
                             attn_depth: int
                             ) -> Tuple[QuantumCircuit,
                                        List[ParameterVector],
                                        List[ParameterVector],
                                        List[SparsePauliOp]]:
    """
    Construct a layered ansatz that mirrors the transformer block structure.
    The circuit is composed of:
        - token encoding (RX gates)
        - a stack of [attention‑like + feed‑forward‑like] blocks
        - measurement operators (Pauli‑Z on each qubit)
    """
    # Encoding
    encoding = _token_encoding(num_qubits)

    # Build block stack
    circuit = QuantumCircuit(num_qubits)
    for _ in range(num_blocks):
        # Attention‑like sub‑layer
        circuit.append(_attention_layer(num_qubits, attn_depth), range(num_qubits))
        # Feed‑forward‑like sub‑layer
        circuit.append(_ffn_layer(num_qubits, ffn_depth), range(num_qubits))

    # Measurement observables
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]

    return circuit, encoding, [], observables  # parameters are grouped inside the circuit

# --------------------------------------------------------------------------- #
# 3. HybridQuantumTransformerClassifier – execution wrapper
# --------------------------------------------------------------------------- #

class HybridQuantumTransformerClassifier:
    """Quantum transformer classifier.

    The forward pass encodes the input into a quantum state, runs the
    parameter‑encoded circuit on a simulator and returns the
    probabilities of measuring |1> on each class qubit.
    """

    def __init__(self,
                 num_qubits: int = 8,
                 num_blocks: int = 4,
                 ffn_depth: int = 1,
                 attn_depth: int = 1,
                 backend: Optional[Aer.backends.base.BaseBackend] = None):
        self.num_qubits = num_qubits
        self.circuit, self.encoding, _, self.observables = build_classifier_circuit(
            num_qubits, num_blocks, ffn_depth, attn_depth
        )
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch, num_qubits) with values in [0, 1] that will be fed to
            the RX encoding gates.

        Returns
        -------
        np.ndarray
            Class probabilities of shape (batch, num_qubits).
        """
        batch_size = inputs.shape[0]
        # Bind parameters for each sample
        param_binds = [{self.encoding[i]: val for i, val in enumerate(sample)}
                       for sample in inputs]

        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        results = job.result()

        probs = np.zeros((batch_size, self.num_qubits))
        for i, obs in enumerate(self.observables):
            # Expectation value of Z = P(0) - P(1)
            exp_val = results.get_expectation_value(obs, self.circuit)
            probs[:, i] = (1 - exp_val) / 2  # convert to probability of |1>
        return probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that accepts a torch tensor."""
        return torch.tensor(self.predict(x.detach().cpu().numpy()), dtype=torch.float32)

__all__ = [
    "HybridQuantumTransformerClassifier",
    "build_classifier_circuit",
    "HybridQuantumTransformerClassifier",
]
