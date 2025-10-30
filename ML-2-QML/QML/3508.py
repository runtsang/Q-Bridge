import torch
import torch.nn as nn
from qiskit import Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.opflow import PauliSumOp
import numpy as np
from typing import Iterable, Tuple

class QuantumNATQuantum(nn.Module):
    """
    Quantum module that accepts a 128‑dim classical embedding,
    performs a 7‑qubit amplitude encoding, runs a parameter‑shift
    variational ansatz, and returns Z‑observable expectations.
    """
    def __init__(self, n_qubits: int = 7, depth: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        # Trainable angles for the variational circuit
        self.theta = nn.Parameter(torch.randn(depth * n_qubits) * 0.1)
        self._circuit = self._build_ansatz()

    def _build_ansatz(self) -> 'QuantumCircuit':
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self.n_qubits)
        for d in range(self.depth):
            for q in range(self.n_qubits):
                qc.ry(self.theta[d * self.n_qubits + q], q)
            for q in range(self.n_qubits - 1):
                qc.cz(q, q + 1)
        return qc

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        """
        embed: (batch, 2**n_qubits) amplitude vector
        returns: (batch, n_qubits) Z expectations
        """
        bsz = embed.shape[0]
        # Normalise embeddings to unit norm
        amps = embed / torch.norm(embed, dim=1, keepdim=True)
        amps_np = amps.detach().cpu().numpy()
        results = []
        for amp in amps_np:
            sv = Statevector(amp)
            sv = sv.evolve(self._circuit)
            z_vals = []
            for i in range(self.n_qubits):
                exp = sv.expectation_value(PauliSumOp.from_list([("Z", [i])]))
                z_vals.append(float(exp))
            results.append(z_vals)
        return torch.tensor(results, dtype=embed.dtype, device=embed.device)

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[object, Iterable, Iterable, list]:
    """
    Construct a quantum classifier ansatz that mirrors the classical
    classifier factory.  Returns the circuit, encoding parameters,
    variational parameters, and a list of Z observables per qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    # Data encoding by RX rotations
    for q in range(num_qubits):
        qc.rx(encoding[q], q)

    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)

    observables = [PauliSumOp.from_list([("Z", [i])]) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

__all__ = ["QuantumNATQuantum", "build_classifier_circuit"]
